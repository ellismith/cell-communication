#!/usr/bin/env Rscript
# run_mashr_ccc.R
# Run mashr on cell-cell communication age coefficients across regions.
# For a given sender or receiver cell type, builds a matrix of:
#   rows    = LR pairs (ligand|receptor)
#   columns = conditions (source_louvain|target_louvain|region if --louvain,
#             else target_celltype|region for sender, source_celltype|region for receiver)
#   values  = age_coef averaged across Louvain subtypes within each condition
# Applies mashr to learn sharing patterns across conditions and shrink estimates.
#
# Usage:
#   Rscript run_mashr_ccc.R --source_ct Astrocyte --louvain --nan_filter 0.55
#   Rscript run_mashr_ccc.R --target_ct GABA --louvain --nan_filter 0.55
#   Rscript run_mashr_ccc.R --source_ct Astrocyte   # cell-type averaged, no louvain

library(mashr)

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# Supports: --source_ct, --target_ct, --nan_filter, --louvain (flag, no value)
args <- commandArgs(trailingOnly=TRUE)

source_ct   <- NULL
target_ct   <- NULL
nan_filter  <- 1.0    # default: keep all conditions regardless of NaN rate
use_louvain <- FALSE  # default: average to cell-type level, not Louvain level

i <- 1
while (i <= length(args)) {
    if (args[i] == "--source_ct")       { source_ct  <- args[i+1]; i <- i+2 }
    else if (args[i] == "--target_ct")  { target_ct  <- args[i+1]; i <- i+2 }
    else if (args[i] == "--nan_filter") { nan_filter <- as.numeric(args[i+1]); i <- i+2 }
    else if (args[i] == "--louvain")    { use_louvain <- TRUE; i <- i+1 }
    else { i <- i+1 }
}

if (is.null(source_ct) && is.null(target_ct)) stop("Must provide --source_ct or --target_ct")
if (!is.null(source_ct) && !is.null(target_ct)) stop("Provide only one of --source_ct or --target_ct")

# Build output label from arguments
mode    <- if (!is.null(source_ct)) "source" else "target"
ct      <- if (mode == "source") source_ct else target_ct
res     <- if (mode == "source") "sender" else "receiver"
lou_tag <- if (use_louvain) "_louvain" else ""
nan_tag <- if (nan_filter < 1.0) paste0("_nanfilt", nan_filter) else ""
label   <- paste0(ct, "_", res, lou_tag, nan_tag)

cat(sprintf("Mode: %s | CT: %s | Louvain: %s | NaN filter: %.2f\n", mode, ct, use_louvain, nan_filter))
cat(sprintf("Label: %s\n", label))

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE    <- "/scratch/easmit31/cell_cell/results/within_region_analysis"
OUT_DIR <- file.path("/scratch/easmit31/cell_cell/results/mashr", label)
REGIONS <- c("ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCb", "M1", "MB", "mdTN", "NAc")
dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)

# ── LOAD REGRESSION CSVS ──────────────────────────────────────────────────────
# Read age_coef and age_stderr for all regions, combine into one data frame
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
df <- do.call(rbind, all_rows)

# ── PARSE INTERACTION STRING ──────────────────────────────────────────────────
# Format: source_louvain|target_louvain|ligand_complex|receptor_complex
# Extract cell type by stripping trailing _N louvain index
parts        <- strsplit(df$interaction, "\\|")
df$source_ct <- sub("_[^_]+$", "", sapply(parts, "[", 1))
df$target_ct <- sub("_[^_]+$", "", sapply(parts, "[", 2))
df$lr_pair   <- paste(sapply(parts, "[", 3), sapply(parts, "[", 4), sep="|")

# Filter to the cell type of interest
if (mode == "source") {
    df <- df[df$source_ct == ct, ]
} else {
    df <- df[df$target_ct == ct, ]
}

# Define conditions: either full Louvain combo x region, or celltype x region
if (use_louvain) {
    # Conditions = source_louvain|target_louvain|region
    p2 <- strsplit(df$interaction, "\\|")
    df$condition <- paste(sapply(p2, "[", 1), sapply(p2, "[", 2), df$region, sep="|")
} else {
    # Conditions = partner_celltype|region (averaged across Louvain subtypes)
    if (mode == "source") {
        df$condition <- paste(df$target_ct, df$region, sep="|")
    } else {
        df$condition <- paste(df$source_ct, df$region, sep="|")
    }
}
cat(sprintf("Filtered to %d rows | %d unique conditions\n", nrow(df), length(unique(df$condition))))

# ── BUILD BHAT AND SHAT MATRICES ──────────────────────────────────────────────
# Average age_coef across Louvain subtypes within each lr_pair x condition cell
# For SE: use RMS (root mean square) rather than mean, which is statistically
# more appropriate when combining standard errors across subtypes
agg_beta <- aggregate(age_coef   ~ lr_pair + condition, data=df, FUN=mean)
agg_se   <- aggregate(age_stderr ~ lr_pair + condition, data=df,
                      FUN=function(x) sqrt(mean(x^2)))

# Pivot to wide format: rows=lr_pairs, cols=conditions
Bhat <- reshape(agg_beta, idvar="lr_pair", timevar="condition", direction="wide")
Shat <- reshape(agg_se,   idvar="lr_pair", timevar="condition", direction="wide")
lr_pairs <- Bhat$lr_pair
Bhat <- as.matrix(Bhat[,-1]); Shat <- as.matrix(Shat[,-1])
rownames(Bhat) <- lr_pairs; rownames(Shat) <- lr_pairs
colnames(Bhat) <- sub("age_coef\\.", "", colnames(Bhat))
colnames(Shat) <- sub("age_stderr\\.", "", colnames(Shat))

cat(sprintf("Matrix before filter: %d LR pairs x %d conditions, NaN=%.1f%%\n",
            nrow(Bhat), ncol(Bhat), 100*mean(is.na(Bhat))))

# ── NaN COLUMN FILTER ─────────────────────────────────────────────────────────
# Drop conditions (columns) where more than nan_filter fraction of LR pairs
# are missing. This removes rare Louvain combos that only appear in a few animals.
# After dropping columns, also drop rows that are now entirely NA.
if (nan_filter < 1.0) {
    col_nan <- colMeans(is.na(Bhat))
    keep    <- col_nan <= nan_filter
    cat(sprintf("NaN filter %.2f: keeping %d/%d conditions\n", nan_filter, sum(keep), length(keep)))
    Bhat <- Bhat[, keep, drop=FALSE]
    Shat <- Shat[, keep, drop=FALSE]
    row_keep <- rowSums(!is.na(Bhat)) > 0
    Bhat <- Bhat[row_keep, , drop=FALSE]
    Shat <- Shat[row_keep, , drop=FALSE]
}

cat(sprintf("Matrix after filter: %d LR pairs x %d conditions, NaN=%.1f%%\n",
            nrow(Bhat), ncol(Bhat), 100*mean(is.na(Bhat))))

# ── IDENTIFY STRONG SIGNALS FOR COVARIANCE ESTIMATION ────────────────────────
# mashr uses a subset of "strong" signals to estimate data-driven covariance
# structures via PCA and ED. We use max |Z| across conditions to rank LR pairs.
# If fewer than 20 pass |Z|>=2, fall back to top 200 by max_z to ensure
# enough rows for stable PCA decomposition.
Z <- Bhat / Shat
Z[is.na(Z)] <- 0
max_z <- apply(abs(Z), 1, max, na.rm=TRUE)
cat(sprintf("|Z|>2: %d, |Z|>3: %d\n", sum(max_z>2), sum(max_z>3)))

strong_idx <- which(max_z >= 2)
if (length(strong_idx) < 20) {
    cat("Fewer than 20 strong signals at |Z|>=2, falling back to top 200 by max_z\n")
    strong_idx <- order(max_z, decreasing=TRUE)[1:min(200, length(max_z))]
}
cat(sprintf("Strong signals used for covariance estimation: %d\n", length(strong_idx)))

# ── FILL MISSING VALUES FOR MASHR ─────────────────────────────────────────────
# mashr requires complete matrices. Fill missing cells with:
#   Bhat = 0 (null effect assumption for missing data)
#   Shat = 1e6 (very large SE = effectively missing, will be shrunk to null)
Bhat_orig <- Bhat; Shat_orig <- Shat
Shat[is.na(Shat)] <- 1e6
Bhat[is.na(Bhat)] <- 0

# ── MASHR: ESTIMATE COVARIANCE STRUCTURES ────────────────────────────────────
# Fit data-driven (ED/PCA) and canonical covariance matrices on strong signals,
# then fit the full mashr model on all LR pairs using the learned mixture.
cat("Setting up mashr data objects...\n")
mash_data        <- mash_set_data(Bhat, Shat)
mash_data_strong <- mash_set_data(Bhat[strong_idx, , drop=FALSE],
                                  Shat[strong_idx, , drop=FALSE])

# npc: number of PCA components for data-driven covariances
# Use min(5, max(1, ncol-1)) to avoid npc=0 when very few conditions
npc    <- min(5, max(1, ncol(Bhat) - 1))
U_pca  <- cov_pca(mash_data_strong, npc=npc)
U_ed   <- cov_ed(mash_data_strong, U_pca)
U_can  <- cov_canonical(mash_data)

cat("Fitting mashr model...\n")
mash_model <- mash(mash_data, Ulist=c(U_ed, U_can), verbose=TRUE)

# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────
cat("\n--- Diagnostics ---\n")
cat(sprintf("Log likelihood: %.2f\n", get_loglik(mash_model)))
cat("Estimated mixture weights (top 10):\n")
pi_est <- sort(get_estimated_pi(mash_model), decreasing=TRUE)
print(head(pi_est, 10))

# ── EXTRACT AND SAVE RESULTS ──────────────────────────────────────────────────
# Save long-format results with posterior estimates and original values
pm   <- get_pm(mash_model)
psd  <- get_psd(mash_model)
lfsr <- get_lfsr(mash_model)

results <- data.frame(
    lr_pair        = rep(rownames(pm), ncol(pm)),
    condition      = rep(colnames(pm), each=nrow(pm)),
    beta_posterior = as.vector(pm),
    sd_posterior   = as.vector(psd),
    lfsr           = as.vector(lfsr),
    # Restore NA for originally missing cells (Shat_orig >= 1e5)
    beta_original  = as.vector(ifelse(Shat_orig >= 1e5, NA, Bhat_orig)),
    se_original    = as.vector(ifelse(Shat_orig >= 1e5, NA, Shat_orig))
)

out_file <- file.path(OUT_DIR, paste0("mashr_", label, "_results.csv"))
write.csv(results, out_file, row.names=FALSE)
saveRDS(mash_model, file.path(OUT_DIR, paste0("mashr_", label, "_model.rds")))
cat(sprintf("\nSaved results: %s\n", out_file))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cat("\n=== SUMMARY ===\n")
cat(sprintf("LR pairs: %d | Conditions: %d\n", nrow(lfsr), ncol(lfsr)))
cat(sprintf("lfsr < 0.05: %d\n", sum(lfsr < 0.05, na.rm=TRUE)))
cat(sprintf("lfsr < 0.10: %d\n", sum(lfsr < 0.10, na.rm=TRUE)))

# Distribution of sharing across conditions
sig_counts <- table(results$lr_pair[results$lfsr < 0.05])
n_conds <- ncol(lfsr)
cat(sprintf("\nSharing distribution (lfsr<0.05):\n"))
cat(sprintf("  sig in all %d conditions: %d\n", n_conds, sum(sig_counts == n_conds)))
cat(sprintf("  sig in >50%%:              %d\n", sum(sig_counts > n_conds/2)))
cat(sprintf("  sig in 6-20:              %d\n", sum(sig_counts >= 6 & sig_counts <= 20)))
cat(sprintf("  sig in 2-5:               %d\n", sum(sig_counts >= 2 & sig_counts <= 5)))
cat(sprintf("  sig in exactly 1:         %d\n", sum(sig_counts == 1)))

cat("\nPer condition (lfsr<0.05):\n")
for (cond in colnames(lfsr)) {
    n <- sum(lfsr[,cond] < 0.05, na.rm=TRUE)
    if (n > 0) cat(sprintf("  %s: %d\n", cond, n))
}
