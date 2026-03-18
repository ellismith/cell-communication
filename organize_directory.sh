#!/bin/bash
# Moves old/superseded scripts to archive/ directory
# Does NOT delete anything

ARCHIVE="/scratch/easmit31/cell_cell/archive"
mkdir -p ${ARCHIVE}

# List of scripts to KEEP in place
KEEP=(
    # CCC pipeline
    "run_per_animal_louvain_v2.py"
    "build_lr_matrix.py"
    "run_age_sex_regression.py"
    # mashr R scripts
    "run_mashr_ccc.R"
    "run_mashr_ccc_topn.R"
    "run_mashr_louvain.R"
    "compare_mashr_covariance.R"
    # test mashr scripts
    "run_mashr_astro_gaba_test.R"
    "run_mashr_astro_gaba_test.sh"
    "run_mashr_astro_glut_test.R"
    "run_mashr_astro_sender_test.R"
    # annotations
    "build_cpdb_annotations.py"
    "cpdb_lr_annotations.csv"
    "lr_functional_annotations.py"
    # mashr tables
    "build_mashr_master_table.py"
    "build_mashr_master_table_top8.py"
    "build_mashr_region_table.py"
    "build_mashr_region_table_top8.py"
    "build_mashr_comparison_tables.py"
    # visualization - regression
    "plot_regression_region_celltype_heatmap.py"
    "plot_regression_region_celltype_heatmap_nanfilt.py"
    "plot_regression_per_region_dotplot.py"
    "plot_mashr_per_region_dotplot_top8.py"
    # visualization - mashr nanfilt
    "plot_mashr_sharing_table.py"
    "plot_mashr_beta_heatmap.py"
    "plot_mashr_beta_heatmap_all.py"
    "plot_mashr_beta_heatmap_bytier.py"
    "plot_mashr_bubble_matrix.py"
    "plot_mashr_category_enrichment.py"
    "plot_mashr_condition_specific.py"
    "plot_mashr_covariance.py"
    "plot_mashr_mixture_weights.py"
    "plot_mashr_regional_dotplot.py"
    "plot_mashr_summary_heatmap.py"
    "plot_mashr_top_lr_pairs.py"
    "plot_mashr_scatter_sanity.py"
    "plot_mashr_cross_sender_scatter.py"
    # visualization - mashr top8
    "plot_mashr_sharing_table_top8.py"
    "plot_mashr_beta_heatmap_top8.py"
    "plot_mashr_beta_heatmap_all_top8.py"
    "plot_mashr_beta_heatmap_bytier_top8.py"
    "plot_mashr_bubble_matrix_top8.py"
    "plot_mashr_category_enrichment_top8.py"
    "plot_mashr_condition_specific_top8.py"
    "plot_mashr_covariance_top8.py"
    "plot_mashr_mixture_weights_top8.py"
    "plot_mashr_regional_dotplot_top8.py"
    "plot_mashr_summary_heatmap_top8.py"
    "plot_mashr_top_lr_pairs_top8.py"
    # submission scripts
    "submit_mashr_ccc.sh"
    "rerun_dlpfc_attempt2.sh"
    "rerun_dlpfc_regression.sh"
    "submit_whole_region_regression.sh"
    "submit_mashr_louvain_test.sh"
    # this script
    "organize_directory.sh"
    # data files
    "cellphonedb_data.zip"
    "cellphonedb_interactions_all_with_complexes.csv"
    "cellphonedb_interactions_expanded_full.csv"
    "cellphonedb_interactions_liana_format.csv"
    "cellchatdb_lr_pairs.csv"
    "top50_log_lrs.txt"
    "top50_raw_lrs.txt"
)

# convert to associative array for fast lookup
declare -A KEEP_MAP
for f in "${KEEP[@]}"; do
    KEEP_MAP[$f]=1
done

# move all .py and .R and .sh files not in KEEP list
moved=0
for f in *.py *.R *.sh; do
    [ -f "$f" ] || continue
    if [ -z "${KEEP_MAP[$f]}" ]; then
        mv "$f" "${ARCHIVE}/"
        echo "  Archived: $f"
        ((moved++))
    fi
done

echo ""
echo "Archived ${moved} files to ${ARCHIVE}/"
echo "Remaining scripts:"
ls *.py *.R *.sh 2>/dev/null

# Archive old directories
ARCHIVE_DIRS=(
    "animal_region_jobs"
    "chord_jobs" "chord_jobs_v2"
    "csv_files"
    "expr_prop_test_jobs"
    "figures"
    "heatmap_jobs" "heatmap_jobs_v2"
    "lm_jobs" "lm_louvain_jobs" "lm_rerun_jobs" "lm_threshold_jobs"
    "lm_plots" "lm_plots_filtering0p0" "lm_plots_filtering0p0_lrmeans"
    "lm_plots_threshold_comparisons" "lm_plots_thresholds"
    "lm_results" "lm_results_filtering0p0" "lm_results_filtering0p0_lrmeans"
    "lm_results_threshold0p0" "lm_results_threshold0p05" "lm_results_threshold0p1"
    "lm_results_threshold0p15" "lm_results_threshold0p2" "lm_results_threshold0p25"
    "lm_results_threshold0p3"
    "louvain_jobs"
    "matrix_jobs"
    "neurogenic_signaling_plots"
    "old"
    "pairwise_jobs" "pairwise_jobs_glut_retry"
    "pairwise_jobs_with_age" "pairwise_jobs_with_age_glut_retry"
    "pairwise_jobs_with_sex" "pairwise_jobs_with_sex_glut_retry"
    "per_animal_louvain_jobs"
    "plotting_scripts"
    "regression_jobs" "regression_jobs_v2" "regression_jobs_v2_age_filtered"
    "regression_jobs_whole_region"
    "sh_files" "txt_files"
    "threshold_summary_plots"
    "u01_all_individuals_jobs" "u01_all_individuals_jobs_filtering0p0"
    "u01_per_individual_jobs" "u01_per_individual_jobs_CPDB5"
    "u01_threshold_test_jobs_threshold0p0" "u01_threshold_test_jobs_threshold0p05"
    "u01_threshold_test_jobs_threshold0p1" "u01_threshold_test_jobs_threshold0p15"
    "u01_threshold_test_jobs_threshold0p2" "u01_threshold_test_jobs_threshold0p25"
    "u01_threshold_test_jobs_threshold0p3"
)

for d in "${ARCHIVE_DIRS[@]}"; do
    if [ -d "$d" ]; then
        mv "$d" "${ARCHIVE}/"
        echo "  Archived dir: $d"
    fi
done

# Archive loose files
for f in qq_plot_validation.png top50_log_lrs.txt top50_raw_lrs.txt run_per_animal_louvain.py.backup; do
    if [ -f "$f" ]; then
        mv "$f" "${ARCHIVE}/"
        echo "  Archived file: $f"
    fi
done
