#!/usr/bin/env Rscript
library(mashr)

MASHR_DIR <- "/scratch/easmit31/cell_cell/results/mashr"
OUT_DIR   <- file.path(MASHR_DIR, "plots_top8perregion")
dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)

CTS <- c("Astrocyte","GABA","Glutamatergic","Microglia","Oligo","OPC",
         "Vascular","Basket","Cerebellar","Ependymal","Midbrain","MSN")

find_nanfilt <- function(ct) {
    for (tag in c("0.55","0.54","0.52","0.50","0.48","0.46")) {
        label <- paste0(ct, "_sender_louvain_nanfilt", tag)
        rds   <- file.path(MASHR_DIR, label, paste0("mashr_", label, "_model.rds"))
        if (file.exists(rds)) return(rds)
    }
    return(NULL)
}

find_top8 <- function(ct) {
    label <- paste0(ct, "_sender_louvain_top8perregion")
    rds   <- file.path(MASHR_DIR, label, paste0("mashr_", label, "_model.rds"))
    if (file.exists(rds)) return(rds)
    return(NULL)
}

get_dominant_cov <- function(model) {
    pi       <- model$fitted_g$pi
    Ulist    <- model$fitted_g$Ulist
    pi_names <- names(pi)

    # match pi names to Ulist names
    ulist_names <- names(Ulist)

    # find ED components that exist in both pi and Ulist
    ed_idx_pi <- grep("ED", pi_names)
    ed_idx_ul <- grep("ED", ulist_names)

    if (length(ed_idx_pi) > 0 && length(ed_idx_ul) > 0) {
        # find highest weight ED component that exists in Ulist
        for (i in ed_idx_pi[order(pi[ed_idx_pi], decreasing=TRUE)]) {
            nm <- pi_names[i]
            ul_i <- which(ulist_names == nm)
            if (length(ul_i) > 0) {
                return(list(U=Ulist[[ul_i[1]]], name=nm, weight=pi[i]))
            }
        }
    }

    # fall back: highest weight component in Ulist excluding null
    non_null <- which(!grepl("null", ulist_names))
    if (length(non_null) > 0) {
        # match to pi
        for (nm in ulist_names[non_null]) {
            pi_i <- which(pi_names == nm)
            if (length(pi_i) > 0 && pi[pi_i] == max(pi[non_null])) {
                ul_i <- which(ulist_names == nm)
                return(list(U=Ulist[[ul_i[1]]], name=nm, weight=pi[pi_i]))
            }
        }
        # just take first non-null
        return(list(U=Ulist[[non_null[1]]], name=ulist_names[non_null[1]], weight=NA))
    }
    return(NULL)
}

cat(sprintf("%-18s %-30s %-30s %s\n", "CT", "nanfilt_top_cov", "top8_top_cov", "shared_corr"))
cat(paste(rep("-", 100), collapse=""), "\n")

results <- list()

for (ct in CTS) {
    rds1 <- find_nanfilt(ct)
    rds2 <- find_top8(ct)
    if (is.null(rds1) || is.null(rds2)) {
        cat(sprintf("%-18s MISSING\n", ct))
        next
    }

    m1 <- readRDS(rds1)
    m2 <- readRDS(rds2)

    cov1 <- get_dominant_cov(m1)
    cov2 <- get_dominant_cov(m2)

    if (is.null(cov1) || is.null(cov2)) {
        cat(sprintf("%-18s could not extract covariance\n", ct))
        next
    }

    conds1 <- colnames(m1$result$PosteriorMean)
    conds2 <- colnames(m2$result$PosteriorMean)
    shared  <- intersect(conds1, conds2)

    corr_val <- NA
    if (length(shared) >= 3) {
        idx1 <- match(shared, conds1)
        idx2 <- match(shared, conds2)
        n1   <- nrow(cov1$U)
        n2   <- nrow(cov2$U)
        if (all(idx1 <= n1) && all(idx2 <= n2)) {
            sub1 <- cov1$U[idx1, idx1, drop=FALSE]
            sub2 <- cov2$U[idx2, idx2, drop=FALSE]
            ut1  <- sub1[upper.tri(sub1)]
            ut2  <- sub2[upper.tri(sub2)]
            if (length(ut1) > 1 && sd(ut1) > 0 && sd(ut2) > 0) {
                corr_val <- cor(ut1, ut2)
            }
        }
    }

    cat(sprintf("%-18s %-30s %-30s %s\n",
                ct,
                paste0(substr(cov1$name,1,25), " (w=", round(cov1$weight,3), ")"),
                paste0(substr(cov2$name,1,25), " (w=", round(cov2$weight,3), ")"),
                ifelse(is.na(corr_val),
                       paste0("n=", length(shared), " shared (insufficient)"),
                       sprintf("r=%.3f (n=%d shared)", corr_val, length(shared)))))

    results[[ct]] <- list(cov1=cov1, cov2=cov2,
                          shared=shared, corr=corr_val,
                          conds1=conds1, conds2=conds2)
}

# ── HEATMAPS ──────────────────────────────────────────────────────────────────
cat("\nSaving covariance comparison heatmaps...\n")

for (ct in names(results)) {
    r <- results[[ct]]
    if (length(r$shared) < 3) next

    idx1 <- match(r$shared, r$conds1)
    idx2 <- match(r$shared, r$conds2)
    if (any(idx1 > nrow(r$cov1$U)) || any(idx2 > nrow(r$cov2$U))) next

    sub1 <- r$cov1$U[idx1, idx1, drop=FALSE]
    sub2 <- r$cov2$U[idx2, idx2, drop=FALSE]
    sub1_norm <- sub1 / max(abs(sub1))
    sub2_norm <- sub2 / max(abs(sub2))

    out <- file.path(OUT_DIR, paste0("covariance_comparison_", ct, "_sender.png"))
    png(out, width=1400, height=650, res=120)
    par(mfrow=c(1,2), mar=c(9,9,3,2))
    cols <- colorRampPalette(c("#4575b4","white","#d73027"))(100)

    image(sub1_norm, col=cols, zlim=c(-1,1), axes=FALSE,
          main=paste0(ct, " nanfilt\n", substr(r$cov1$name,1,30)))
    axis(1, at=seq(0,1,length.out=length(r$shared)),
         labels=r$shared, las=2, cex.axis=0.45)
    axis(2, at=seq(0,1,length.out=length(r$shared)),
         labels=r$shared, las=2, cex.axis=0.45)

    image(sub2_norm, col=cols, zlim=c(-1,1), axes=FALSE,
          main=paste0(ct, " top8perregion\n", substr(r$cov2$name,1,30)))
    axis(1, at=seq(0,1,length.out=length(r$shared)),
         labels=r$shared, las=2, cex.axis=0.45)
    axis(2, at=seq(0,1,length.out=length(r$shared)),
         labels=r$shared, las=2, cex.axis=0.45)

    dev.off()
    cat(sprintf("  Saved: %s\n", out))
}
