"""Shared susceptibility workflow: train/evaluate -> map -> classes -> figures -> report sections."""
import numpy as np

from . import config as C, ml, viz


def run(tag, arr, prof, names, X, y, groups, factors, hazard, report, models=None, points=None,
        point_label="Inventory"):
    out = C.out_dir(tag)
    models = models or ml.make_models()
    metrics, fitted, (te, preds) = ml.evaluate(models, X, y, groups)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    roc = ml.plot_roc(y[te], preds, out / "roc.png")
    best = metrics.sort_values("AUC_test", ascending=False).iloc[0]["model"]
    imp = ml.importance(fitted[best], X.iloc[te], y[te], out / "importance.png",
                        title=f"Factor importance - {best}")
    imp.to_csv(out / "factor_importance.csv", index=False)
    prob = ml.predict_map(fitted[best], arr, names, factors)
    cls, br = ml.classify_quantiles(prob)
    ml.save_raster(prob, prof, out / f"{hazard}_susceptibility.tif")
    ml.save_raster(cls, prof, out / f"{hazard}_susceptibility_class.tif")
    b = ml.bounds_of(prof)
    viz.plot_raster(prob, b, f"{hazard} susceptibility ({best})", out / "map_probability.png",
                    cmap="RdYlBu_r", vmin=0, vmax=1, label="Probability", points=points, point_label=point_label)
    viz.plot_classes(cls, b, f"{hazard} susceptibility classes", out / "map_classes.png", points=points,
                     point_label=point_label)
    area = ml.class_area_table(cls, prof)
    area.to_csv(out / "class_area.csv", index=False)

    report.h("Results", 1)
    report.p(f"Models were evaluated on a spatially separated hold-out (30 % of 5 km blocks) and 5-fold "
             f"spatial-block cross-validation, which avoids the optimistic bias of random splits caused by "
             f"spatial autocorrelation. The best model was {best} (AUC = {metrics.set_index('model').loc[best, 'AUC_test']:.3f}).")
    report.table(metrics, "Model performance")
    report.fig(out / "roc.png", "ROC curves on the spatial hold-out", 4.6)
    report.fig(out / "importance.png", "Permutation importance of conditioning factors", 5.2)
    report.table(imp.head(10), "Ten most important factors")
    report.fig(out / "map_probability.png", f"{hazard} susceptibility probability", 5.8)
    report.fig(out / "map_classes.png", f"{hazard} susceptibility classes (quantile breaks)", 5.8)
    report.table(area, "Area per susceptibility class")
    return dict(metrics=metrics, best=best, fitted=fitted, prob=prob, cls=cls, imp=imp, area=area, out=out)
