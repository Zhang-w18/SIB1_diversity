from sib1div.analysis.plan005 import select_full_snr_grid
from sib1div.analysis.plan005_tail import select_tail_plan


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


def test_plan005_tail_selects_guarded_grid_and_per_snr_drop_counts():
    prescan = []
    patterns = {
        "B-SSB": [20, 8, 2, 1], "P2-SSB": [12, 4, 1, 0],
        "P6-SSB": [14, 5, 1, 0], "BC-SSB": [16, 6, 2, 0],
        "CDD-SSB": [8, 3, 1, 0],
    }
    for curve, errors in patterns.items():
        for snr, count in zip((1.0, 1.5, 2.0, 2.5), errors):
            prescan.append({
                "curve_id": curve, "curve_type": "estimated_csi", "snr_db": str(snr),
                "block_errors": str(count), "blocks": "500", "bler": str(count / 500),
            })
    existing = [{
        "curve_id": curve, "curve_type": "estimated_csi", "snr_db": "0.5",
        "block_errors": "30", "blocks": "1000", "bler": "0.03",
    } for curve in patterns]
    grid, brackets, totals = select_tail_plan(prescan, existing)
    assert grid == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert brackets["B-SSB"] == [1.5, 2.0]
    assert brackets["CDD-SSB"] == [1.0, 1.5]
    assert set(totals) == set(grid)
    assert all(value % 1000 == 0 and 2000 <= value <= 20000 for value in totals.values())


def test_plan005_tail_uses_first_crossing_when_sparse_tail_recrosses():
    rows = []
    for curve in ("B-SSB", "P2-SSB", "P6-SSB", "BC-SSB", "CDD-SSB"):
        for snr, errors in ((0.5, 3), (1.0, 2), (1.5, 0), (2.0, 1), (2.5, 0)):
            rows.append({
                "curve_id": curve, "curve_type": "estimated_csi", "snr_db": str(snr),
                "block_errors": str(errors), "blocks": "100", "bler": str(errors / 100),
            })
    grid, brackets, _ = select_tail_plan(rows, [])
    assert brackets["B-SSB"] == [1.0, 1.5]
    assert grid == [0.5, 1.0, 1.5, 2.0]
