
import time
import numpy as np


def test_metrics():
    from hipose.metrics import MetricsAnalyser
    from hipose.metrics import (QADistMetric, RelQADistMetric,
                                DistMetric, RelDistMetric, ProcrustesDistMetric,
                                AUCMetric, PCPMetric, TimeMetric)

    from hipose.rotations import quat_random

    # define metrics to evaluate (between ergowear and xsens skeletons)
    metrics_log = MetricsAnalyser(
            metrics=dict(
                    angl_dist=QADistMetric("angl_dist", show_degrees=True),
                    rel_angl_dist=RelQADistMetric("rel_angl_dist", rel_idx=0),
                    auc_angl_dist=AUCMetric("auc_angl_dist", pcp_thresh_range=(0, np.pi), dist="qad"),
                    pcp_angl_dist=PCPMetric("pcp_angl_dist", dist="qad", threshold=np.pi/6),
                    pos_dist=DistMetric("pos_dist", dist="rmse", units="m"),
                    rel_pos_dist=RelDistMetric("rel_pos_dist", dist="mse", rel_idx=0, units="m"),
                    procrustes_pos_dist=ProcrustesDistMetric("procrustes_pos_dist", dist="l2norm", units="m"),
                    auc_pos_dist=AUCMetric("auc_pos_dist", pcp_thresh_range=(0, 0.150), dist="mae"),
                    pcp_pos_dist=PCPMetric("pcp_pos_dist", dist="rmse", threshold=0.5),
                    processing_time=TimeMetric("processing_time", units="ms"),
            )
    )
    metrics_log.reset()

    nseq = 100
    nsegs = 9

    q1 = quat_random(num=nseq*nsegs).reshape(nseq, nsegs, 4)
    q2 = quat_random(num=nseq*nsegs).reshape(nseq, nsegs, 4)

    prange = (-2., 2.)
    p1 = np.random.rand(nseq, nsegs, 3) * (prange[1] - prange[0]) + prange[0]
    p2 = np.random.rand(nseq, nsegs, 3) * (prange[1] - prange[0]) + prange[0]

    # temporal loop
    for t in range(nseq):
        stime = time.perf_counter()

        # compute metrics
        metrics_log.update(
                dict(
                        # angle
                        angl_dist=[q1[t], q2[t]],
                        rel_angl_dist=[q1[t], q2[t]],
                        auc_angl_dist=[q1[t], q2[t]],
                        pcp_angl_dist=[q1[t], q2[t]],
                        # position
                        pos_dist=[p1[t], p2[t]],
                        rel_pos_dist=[p1[t], p2[t]],
                        procrustes_pos_dist=[p1[t], p2[t]],
                        auc_pos_dist=[p1[t], p2[t]],
                        pcp_pos_dist=[p1[t], p2[t]],
                        processing_time=[(time.perf_counter() - stime) * 1000.],
                )
        )

    metrics_log.log_all(save_path=None, show_plots=False, print_metrics=False)
    metrics = metrics_log.get_metrics()
    avgmetrics = metrics_log.get_avg_metrics()
