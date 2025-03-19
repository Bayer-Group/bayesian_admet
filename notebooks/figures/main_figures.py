import dili_predict as dp


dp.plots.model_comparison_figure(diagnostic="MCC")
dp.plots.logistic_comparison_figure(diagnostic="MCC")

dp.plots.modality_comparison_figure(diagnostic="MCC")
dp.plots.modality_comparison_figure(diagnostic="Precision")
dp.plots.modality_comparison_figure(diagnostic="Recall")

dp.plots.increase_observations_figure(diagnostic="MCC")
dp.plots.increase_observations_figure(diagnostic="Precision")
dp.plots.increase_observations_figure(diagnostic="Recall")

dp.plots.fusion_comparison_figure(diagnostic="MCC")
dp.plots.fusion_comparison_figure(diagnostic="Precision")
dp.plots.fusion_comparison_figure(diagnostic="Recall")

dp.plots.dili_probability_figure("atorvastatin")
dp.plots.dili_probability_figure("acetaminophen")
dp.plots.dili_probability_figure("perhexilene")
dp.plots.dili_probability_figure("chlorpromazine")
