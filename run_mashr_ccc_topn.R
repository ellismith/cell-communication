#!/usr/bin/env Rscript
# run_mashr_ccc_topn.R
# Variant of run_mashr_ccc.R that selects top N densest conditions per region
# instead of filtering by NaN rate threshold. Guarantees regional balance.
#
# Usage:
#   Rscript run_mashr_ccc_topn.R --source_ct Astrocyte --top_n 8
#   Rscript run_mashr_ccc_topn.R --target_ct GABA --top_n 8

library(mashr)

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly=TRUE)

source_ct   <- NULL
target_ct   <- NULL
top_n       <- 8
use_louvain <- TRUE  # always louvain for this script

i <- 1
while (i <= length(args)) {
    if (args[i] == "--source_ct")        { source_ct <- args[i+1]; i <- i+2 }
    else if (args[i] == "--target_ct")   { target_ct <- args[i+1]; i <- i+2 }
    else if (args[i] == "--top_n")       { top_n     <- as.integer(args[i+1]); i <- i+2 }
    else { i <- i+1 }
}

if (is.null(source_ct) && is.null(target_ct)) stop("Must provide --source_ct or --target_ct")
if (!is.null(source_ct) && !is.null(target_ct)) stop("Provide only one of --source_ct or --target_ct")

mode  <- if (!is.null(source_ct)) "source" else "target"
ct    <- if (mode == "source") source_ct else target_ct
res   <- if (mode == "source") "sender" else "receiver"
label <- paste0(ct, "_", res, "_louvain_top", top_n, "perregion")

cat(sprintf("Mode: %s | CT: %s | Top N per region: %d\n", mode, ct, top_n))
cat(sprintf("Label: %s\n", label))

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    <- "/scratch/easmit31/cell_cell/results/within_region_analysis"
OUT_DIR <- file.path("/scratch/easmit31/cell_cell/results/mashr", label)
REGIONS <- c("ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCb", "M1", "MB", "mdTN", "NAc")
dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)

# ── LOAD ──────────────────────────────────────────────────────────────────────
cat("Loading regression CSVs...\n")
all_rows <- list()
for (region in REGIONS) {
    rlow <- tolower(region)
    path <- file.path(BASE, paste0("regression_", region),
                      paste0("whole_", rlow, "_age_sex_regression.csv"))
    if (!file.exists(path)) { cat(sprintf("  MISSING: %s\n", region)); next }
    df <- read.csv(path, stringsAsFactors=FALSE)[, c("interaction","age_coef","age_stderr")]
    df$region <- region
    all_rows[[region]] <- df
    cat(sprintf("  loaded %s\n", region))
}
if (length(all_rows) == 0) stop("No regression CSVs found")
df <- do.call(rbind, all_rows)

# ── PARSE ─────────────────────────────────────────────────────────────────────
parts        <- strsplit(df$interaction, "\\|")
df$source_ct <- sub("_[^_]+$", "", sapply(parts, "[", 1))
df$target_ct <- sub("_[^_]+$", "", sapply(parts, "[", 2))
df$lr_pair   <- paste(sapply(parts, "[", 3), sapply(parts, "[", 4), sep="|")

if (mode == "source") {
    df <- df[df$source_ct == ct, ]
} else {
    df <- df[df$target_ct == ct, ]
}
if (nrow(df) == 0) stop(sprintf("No rows for %s = %s", mode, ct))

p2 <- strsplit(df$interaction, "\\|")
df$condition <- paste(sapply(p2, "[", 1), sapply(p2, "[", 2), df$region, sep="|")
cat(sprintf("Filtered to %d rows | %d unique conditions\n", nrow(df), length(unique(df$condition))))

# ── BUILD FULL MATRIX ─────────────────────────────────────────────────────────
agg_beta <- aggregate(age_coef   ~ lr_pair + condition, data=df, FUN=mean)
agg_se   <- aggregate(age_stderr ~ lr_pair + condition, data=df,
                      FUN=function(x) sqrt(mean(x^2)))
Bhat <- reshape(agg_beta, idvar="lr_pair", timevar="condition", direction="wide")
Shat <- reshape(agg_se,   idvar="lr_pair", timevar="condition", direction="wide")
lr_pairs <- Bhat$lr_pair
Bhat <- as.matrix(Bhat[,-1]); Shat <- as.matrix(Shat[,-1])
rownames(Bhat) <- lr_pairs; rownames(Shat) <- lr_pairs
colnames(Bhat) <- sub("age_coef\\.", "", colnames(Bhat))
colnames(Shat) <- sub("age_stderr\\.", "", colnames(Shat))

cat(sprintf("Full matrix: %d LR pairs x %d conditions\n", nrow(Bhat), ncol(Bhat)))

# ── TOP N PER REGION SELECTION ────────────────────────────────────────────────
# For each region, select the top_n conditions with lowest NaN rate (densest).
# This guarantees regional balance regardless of absolute sparsity.
col_nan    <- colMeans(is.na(Bhat))
cond_names <- colnames(Bhat)
# extract region from condition name (last element after |)
cond_regions <- sapply(strsplit(cond_names, "\\|"), function(x) x[length(x)])

selected <- c()
for (region in REGIONS) {
    idx <- which(cond_regions == region)
    if (length(idx) == 0) {
        cat(sprintf("  %s: no conditions found\n", region))
        next
    }
    # rank by NaN rate ascending, take top_n
    ranked <- idx[order(col_nan[idx])]
    take   <- ranked[1:min(top_n, length(ranked))]
    selected <- c(selected, take)
    cat(sprintf("  %s: %d conditions selected (NaN range: %.1f%% - %.1f%%)\n",
                region, length(take),
                100*min(col_nan[take]), 100*max(col_nan[take])))
}

cat(sprintf("Total selected conditions: %d\n", length(selected)))
Bhat <- Bhat[, selected, drop=FALSE]
Shat <- Shat[, selected, drop=FALSE]

# drop rows now entirely NA
row_keep <- rowSums(!is.na(Bhat)) > 0
Bhat <- Bhat[row_keep, , drop=FALSE]
Shat <- Shat[row_keep, , drop=FALSE]

cat(sprintf("Matrix after selection: %d LR pairs x %d conditions, NaN=%.1f%%\n",
            nrow(Bhat), ncol(Bhat), 100*mean(is.na(Bhat))))

# ── STRONG SIGNALS ────────────────────────────────────────────────────────────
Z <- Bhat / Shat
Z[is.na(Z)] <- 0
max_z <- apply(abs(Z), 1, max, na.rm=TRUE)
cat(sprintf("|Z|>2: %d, |Z|>3: %d\n", sum(max_z>2), sum(max_z>3)))

strong_idx <- which(max_z >= 2)
if (length(strong_idx) < 20) {
    cat("Fewer than 20 strong signals, falling back to top 200\n")
    strong_idx <- order(max_z, decreasing=TRUE)[1:min(200, length(max_z))]
}
cat(sprintf("Strong signals: %d\n", length(strong_idx)))

# ── FILL MISSING ──────────────────────────────────────────────────────────────
Bhat_orig    <- Bhat
Shat_orig    <- Shat
missing_mask <- is.na(Bhat_orig)
Bhat[missing_mask] <- 0
Shat[missing_mask] <- 1e6

# ── MASHR ─────────────────────────────────────────────────────────────────────
cat("Setting up mashr...\n")
mash_data        <- mash_set_data(Bhat, Shat)
mash_data_strong <- mash_set_data(Bhat[strong_idx, , drop=FALSE],
                                  Shat[strong_idx, , drop=FALSE])
npc   <- min(5, max(1, ncol(Bhat) - 1))
U_pca <- cov_pca(mash_data_strong, npc=npc)
U_ed  <- cov_ed(mash_data_strong, U_pca)
U_can <- cov_canonical(mash_data)

cat("Fitting mashr model...\n")
mash_model <- mash(mash_data, Ulist=c(U_ed, U_can), verbose=TRUE)

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
cat("\n--- Diagnostics ---\n")
cat(sprintf("Log likelihood: %.2f\n", get_loglik(mash_model)))
cat("Estimated mixture weights (top 10):\n")
pi_est <- sort(get_estimated_pi(mash_model), decreasing=TRUE)
print(head(pi_est, 10))

# ── RESULTS ───────────────────────────────────────────────────────────────────
pm   <- get_pm(mash_model)
psd  <- get_psd(mash_model)
lfsr <- get_lfsr(mash_model)

results <- data.frame(
    lr_pair        = rep(rownames(pm), ncol(pm)),
    condition      = rep(colnames(pm), each=nrow(pm)),
    beta_posterior = as.vector(pm),
    sd_posterior   = as.vector(psd),
    lfsr           = as.vector(lfsr),
    beta_original  = as.vector(ifelse(missing_mask, NA, Bhat_orig)),
    se_original    = as.vector(ifelse(missing_mask, NA, Shat_orig))
)

out_file <- file.path(OUT_DIR, paste0("mashr_", label, "_results.csv"))
write.csv(results, out_file, row.names=FALSE)
saveRDS(mash_model, file.path(OUT_DIR, paste0("mashr_", label, "_model.rds")))
cat(sprintf("\nSaved: %s\n", out_file))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cat("\n=== SUMMARY ===\n")
cat(sprintf("LR pairs: %d | Conditions: %d\n", nrow(lfsr), ncol(lfsr)))
cat(sprintf("lfsr < 0.05: %d\n", sum(lfsr < 0.05, na.rm=TRUE)))
cat(sprintf("lfsr < 0.10: %d\n", sum(lfsr < 0.10, na.rm=TRUE)))

sig_counts <- table(results$lr_pair[results$lfsr < 0.05])
n_conds <- ncol(lfsr)
cat(sprintf("\nSharing distribution (lfsr<0.05):\n"))
cat(sprintf("  sig in all %d conditions: %d\n", n_conds, sum(sig_counts == n_conds)))
cat(sprintf("  sig in >50%%:              %d\n", sum(sig_counts > n_conds/2)))
cat(sprintf("  sig in 6-20:              %d\n", sum(sig_counts >= 6 & sig_counts <= 20)))
cat(sprintf("  sig in 2-5:               %d\n", sum(sig_counts >= 2 & sig_counts <= 5)))
cat(sprintf("  sig in exactly 1:         %d\n", sum(sig_counts == 1)))

cat("\nConditions per region:\n")
cond_regs <- sapply(strsplit(colnames(lfsr), "\\|"), function(x) x[length(x)])
print(table(cond_regs))

cat("\nPer condition (lfsr<0.05, top 20):\n")
cond_sig <- sort(colSums(lfsr < 0.05, na.rm=TRUE), decreasing=TRUE)
print(head(cond_sig, 20))
