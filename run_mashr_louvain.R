#!/usr/bin/env Rscript
library(mashr)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) {
    cat("Usage: Rscript run_mashr_louvain.R SOURCE_LOUVAIN TARGET_LOUVAIN\n")
    cat("  e.g. Rscript run_mashr_louvain.R Astrocyte_2 Microglia_4\n")
    quit(status=1)
}

source_lou <- args[1]
target_lou <- args[2]
label <- paste0(source_lou, "_to_", target_lou)

BASE    <- "/scratch/easmit31/cell_cell/results/within_region_analysis"
OUT_DIR <- file.path("/scratch/easmit31/cell_cell/results/mashr_louvain", label)
REGIONS <- c("ACC", "CN", "dlPFC", "EC", "HIP", "IPP", "lCb", "M1", "MB", "mdTN", "NAc")
dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)

cat(sprintf("Louvain pair: %s -> %s\n", source_lou, target_lou))

# ── LOAD ──────────────────────────────────────────────────────────────────────
all_rows <- list()
for (region in REGIONS) {
    rlow <- tolower(region)
    path <- file.path(BASE, paste0("regression_", region),
                      paste0("whole_", rlow, "_age_sex_regression.csv"))
    if (!file.exists(path)) next
    df <- read.csv(path, stringsAsFactors=FALSE)[, c("interaction","age_coef","age_stderr")]
    df$region <- region
    all_rows[[region]] <- df
}
df <- do.call(rbind, all_rows)

# ── FILTER TO EXACT LOUVAIN COMBO ─────────────────────────────────────────────
parts      <- strsplit(df$interaction, "\\|")
df$source  <- sapply(parts, "[", 1)
df$target  <- sapply(parts, "[", 2)
df$lr_pair <- paste(sapply(parts, "[", 3), sapply(parts, "[", 4), sep="|")
df <- df[df$source == source_lou & df$target == target_lou, ]
cat(sprintf("Filtered rows: %d\n", nrow(df)))

if (nrow(df) == 0) { cat("No data found, exiting.\n"); quit(status=1) }

# ── BUILD MATRIX: rows=LR pairs, cols=regions ─────────────────────────────────
agg_beta <- aggregate(age_coef   ~ lr_pair + region, data=df, FUN=mean)
agg_se   <- aggregate(age_stderr ~ lr_pair + region, data=df, FUN=mean)
Bhat <- reshape(agg_beta, idvar="lr_pair", timevar="region", direction="wide")
Shat <- reshape(agg_se,   idvar="lr_pair", timevar="region", direction="wide")
lr_pairs <- Bhat$lr_pair
Bhat <- as.matrix(Bhat[,-1]); Shat <- as.matrix(Shat[,-1])
rownames(Bhat) <- lr_pairs; rownames(Shat) <- lr_pairs
colnames(Bhat) <- sub("age_coef\\.", "", colnames(Bhat))
colnames(Shat) <- sub("age_stderr\\.", "", colnames(Shat))

cat(sprintf("Matrix: %d LR pairs x %d regions\n", nrow(Bhat), ncol(Bhat)))
cat(sprintf("NaN rate: %.1f%%\n", 100*mean(is.na(Bhat))))

Z <- Bhat / Shat
Z[is.na(Z)] <- 0
max_z <- apply(abs(Z), 1, max, na.rm=TRUE)
cat(sprintf("|Z|>2: %d, |Z|>3: %d\n", sum(max_z>2), sum(max_z>3)))

if (sum(max_z >= 2) < 5) {
    cat("WARNING: fewer than 5 strong signals, results may be unreliable\n")
}

# ── FILL MISSING ──────────────────────────────────────────────────────────────
Bhat_orig <- Bhat; Shat_orig <- Shat
Shat[is.na(Shat)] <- 1e6
Bhat[is.na(Bhat)] <- 0

# ── MASHR ─────────────────────────────────────────────────────────────────────
mash_data  <- mash_set_data(Bhat, Shat)
strong_idx <- which(max_z >= 2)
if (length(strong_idx) < 5) strong_idx <- which(max_z >= 1.0)
cat(sprintf("Strong signals: %d\n", length(strong_idx)))

mash_data_strong <- mash_set_data(Bhat[strong_idx, , drop=FALSE], Shat[strong_idx, , drop=FALSE])
U_pca <- cov_pca(mash_data_strong, npc=min(3, ncol(Bhat)-1))
U_ed  <- cov_ed(mash_data_strong, U_pca)
U_can <- cov_canonical(mash_data)

cat("Fitting mashr...\n")
mash_model <- mash(mash_data, Ulist=c(U_ed, U_can), verbose=TRUE)

# ── RESULTS ───────────────────────────────────────────────────────────────────
pm   <- get_pm(mash_model)
psd  <- get_psd(mash_model)
lfsr <- get_lfsr(mash_model)

results <- data.frame(
    lr_pair        = rep(rownames(pm), ncol(pm)),
    region         = rep(colnames(pm), each=nrow(pm)),
    beta_posterior = as.vector(pm),
    sd_posterior   = as.vector(psd),
    lfsr           = as.vector(lfsr),
    beta_original  = as.vector(ifelse(Shat_orig >= 1e5, NA, Bhat_orig)),
    se_original    = as.vector(ifelse(Shat_orig >= 1e5, NA, Shat_orig))
)

out_file <- file.path(OUT_DIR, paste0("mashr_", label, "_results.csv"))
write.csv(results, out_file, row.names=FALSE)
saveRDS(mash_model, file.path(OUT_DIR, paste0("mashr_", label, "_model.rds")))
cat(sprintf("Saved: %s\n", out_file))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
cat("\n=== SUMMARY ===\n")
cat(sprintf("LR pairs: %d | Regions: %d\n", nrow(lfsr), ncol(lfsr)))
cat(sprintf("lfsr < 0.05: %d\n", sum(lfsr < 0.05, na.rm=TRUE)))
cat(sprintf("lfsr < 0.10: %d\n", sum(lfsr < 0.10, na.rm=TRUE)))
cat("\nPer region (lfsr<0.05):\n")
for (reg in colnames(lfsr)) {
    n <- sum(lfsr[,reg] < 0.05, na.rm=TRUE)
    if (n > 0) cat(sprintf("  %s: %d\n", reg, n))
}
w <- sort(mash_model$fitted_g$pi, decreasing=TRUE)
cat("\nTop mixture weights:\n")
print(head(w, 8))
