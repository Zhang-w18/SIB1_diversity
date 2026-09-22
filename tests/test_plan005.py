from sib1div.analysis.plan005 import select_full_snr_grid


def test_plan005_full_grid_is_guarded_union_of_curve_brackets():
    rows = []
    for curve, values in {
        "B": [(-6, .4), (-4, .08), (-2, .01)],
        "CDD": [(-6, .7), (-4, .2), (-2, .04)],
    }.items():
        rows.extend({
            "curve_id": curve, "curve_type": "estimated_csi",
            "snr_db": str(snr), "bler": str(bler),
        } for snr, bler in values)
    grid, brackets = select_full_snr_grid(rows)
    assert brackets == {"B": [-6.0, -4.0], "CDD": [-4.0, -2.0]}
    assert grid == [-6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5]
