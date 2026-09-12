"""Seasonal-Naive normalised aggregate over GIFT-Eval configs.

GIFT-Eval reports one MASE / CRPS pair per (dataset, term) config. The headline
number used throughout this project is the geometric mean of those values after
dividing each by the Seasonal-Naive baseline for the same config, so that
datasets on wildly different scales contribute equally.

SN_DATA holds the Seasonal-Naive reference values; read_metrics() consumes the
CSV produced by evaluate.py and returns (geo-mean MASE, geo-mean CRPS, n).
"""
import csv
import math

SN_DATA = {
    "bitbrains_fast_storage/5T/long": (1.136642232, 1.177307703),
    "bitbrains_fast_storage/5T/medium": (1.22027136, 1.197912898),
    "bitbrains_fast_storage/5T/short": (1.136024174, 1.210386425),
    "bitbrains_fast_storage/H/short": (1.298512457, 1.022239552),
    "bitbrains_rnd/5T/long": (3.501257904, 1.175220507),
    "bitbrains_rnd/5T/medium": (4.542392315, 1.1692913),
    "bitbrains_rnd/5T/short": (1.970812638, 1.101590444),
    "bitbrains_rnd/H/short": (6.037361869, 1.243217095),
    "bizitobs_application/10S/long": (3.206345056, 0.045730902),
    "bizitobs_application/10S/medium": (2.691418339, 0.042689124),
    "bizitobs_application/10S/short": (2.242330239, 0.034838682),
    "bizitobs_l2c/5T/long": (1.454283044, 0.648492127),
    "bizitobs_l2c/5T/medium": (1.243599799, 0.520391775),
    "bizitobs_l2c/5T/short": (0.986021081, 0.262068386),
    "bizitobs_l2c/H/long": (1.426054153, 0.941065124),
    "bizitobs_l2c/H/medium": (1.510286123, 0.904205135),
    "bizitobs_l2c/H/short": (1.214064127, 0.521167577),
    "bizitobs_service/10S/long": (1.36719347, 0.053452996),
    "bizitobs_service/10S/medium": (1.320579073, 0.047558313),
    "bizitobs_service/10S/short": (1.225305043, 0.03998234),
    "car_parts/M/short": (1.201463839, 1.721743894),
    "covid_deaths/D/short": (46.91239826, 0.126723278),
    "electricity/15T/long": (1.163531708, 0.112553324),
    "electricity/15T/medium": (1.150788204, 0.112754853),
    "electricity/15T/short": (1.717060141, 0.164892615),
    "electricity/D/short": (1.9870414, 0.104059871),
    "electricity/H/long": (1.524078616, 0.153418398),
    "electricity/H/medium": (1.392453785, 0.127387012),
    "electricity/H/short": (1.357772605, 0.105664995),
    "electricity/W/short": (2.089699506, 0.099303202),
    "ett1/15T/long": (1.190885706, 0.340179347),
    "ett1/15T/medium": (1.188305978, 0.321551169),
    "ett1/15T/short": (0.934154959, 0.241363949),
    "ett1/D/short": (1.778369788, 0.408433578),
    "ett1/H/long": (1.478701287, 0.47106249),
    "ett1/H/medium": (1.567800746, 0.43491183),
    "ett1/H/short": (0.977317624, 0.240368441),
    "ett1/W/short": (1.768920788, 0.311825147),
    "ett2/15T/long": (1.012657775, 0.132824448),
    "ett2/15T/medium": (1.05123915, 0.124128778),
    "ett2/15T/short": (1.0673148, 0.096384388),
    "ett2/D/short": (1.390114316, 0.153311002),
    "ett2/H/long": (1.12840167, 0.207906332),
    "ett2/H/medium": (1.238424973, 0.186205081),
    "ett2/H/short": (0.923167828, 0.088952892),
    "ett2/W/short": (0.778520953, 0.133677589),
    "hierarchical_sales/D/short": (1.1348034, 1.736457207),
    "hierarchical_sales/W/short": (1.025013378, 0.832226753),
    "hospital/M/short": (0.920527827, 0.062488177),
    "jena_weather/10T/long": (0.76149237, 0.237194913),
    "jena_weather/10T/medium": (0.716010039, 0.21183609),
    "jena_weather/10T/short": (0.742887468, 0.155200924),
    "jena_weather/D/short": (1.57341403, 0.210590857),
    "jena_weather/H/long": (1.267717847, 0.419143401),
    "jena_weather/H/medium": (0.888613341, 0.342805522),
    "jena_weather/H/short": (0.722910219, 0.154207888),
    "kdd_cup_2018/D/short": (1.497034374, 0.674506129),
    "kdd_cup_2018/H/long": (1.335533666, 0.936283335),
    "kdd_cup_2018/H/medium": (1.428999707, 0.758789203),
    "kdd_cup_2018/H/short": (1.340425279, 0.54753658),
    "loop_seattle/5T/long": (1.250685754, 0.127166561),
    "loop_seattle/5T/medium": (1.153285147, 0.11727938),
    "loop_seattle/5T/short": (0.762339712, 0.080832026),
    "loop_seattle/D/short": (1.732395706, 0.10325476),
    "loop_seattle/H/long": (1.546026875, 0.187094502),
    "loop_seattle/H/medium": (1.480512619, 0.162080907),
    "loop_seattle/H/short": (1.292841261, 0.104235084),
    "m_dense/D/short": (1.669336881, 0.226865657),
    "m_dense/H/long": (1.477917874, 0.41895079),
    "m_dense/H/medium": (1.569946512, 0.377383599),
    "m_dense/H/short": (1.487544417, 0.274797869),
    "m4_daily/D/short": (3.278424297, 0.024356148),
    "m4_hourly/H/short": (1.193210188, 0.037572551),
    "m4_monthly/M/short": (1.259717039, 0.121920637),
    "m4_quarterly/Q/short": (1.602247175, 0.098053975),
    "m4_weekly/W/short": (2.777295047, 0.060870395),
    "m4_yearly/A/short": (3.965954932, 0.137145657),
    "restaurant/D/short": (1.006075745, 0.677016475),
    "saugeen/D/short": (3.413049306, 0.585014174),
    "saugeen/M/short": (0.976374173, 0.445084629),
    "saugeen/W/short": (1.990658005, 0.734035221),
    "solar/10T/long": (0.870931796, 0.673849484),
    "solar/10T/medium": (0.92700054, 0.655063554),
    "solar/10T/short": (1.105702806, 0.8595387),
    "solar/D/short": (1.155861191, 0.559065116),
    "solar/H/long": (1.071196648, 1.077976418),
    "solar/H/medium": (0.934968591, 0.945987467),
    "solar/H/short": (0.951936246, 0.591655963),
    "solar/W/short": (1.470350678, 0.209747628),
    "sz_taxi/15T/long": (0.690943134, 0.427970406),
    "sz_taxi/15T/medium": (0.713484054, 0.379172159),
    "sz_taxi/15T/short": (0.764416774, 0.308706851),
    "sz_taxi/H/short": (0.738167299, 0.213824941),
    "temperature_rain/D/short": (2.011535765, 1.26794552),
    "us_births/D/short": (1.864839191, 0.119526866),
    "us_births/M/short": (0.760531079, 0.016825597),
    "us_births/W/short": (1.563420434, 0.019296796),
}

def read_metrics(csv_path):
    """
    Read MASE/CRPS from the evaluation CSV, divide each by the Seasonal-Naive
    baseline, and take the geometric mean: exp(mean(ln(model / sn))).
    """
    log_mase, log_crps = [], []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            di = 0
            mi = header.index("MASE[0.5]") if "MASE[0.5]" in header else 6
            ci = (header.index("mean_weighted_sum_quantile_loss")
                  if "mean_weighted_sum_quantile_loss" in header else 13)
            for row in reader:
                try:
                    ds = row[di].strip()
                    m, c = float(row[mi]), float(row[ci])
                    if ds in SN_DATA:
                        sn_m, sn_c = SN_DATA[ds]
                        if sn_m > 0 and m > 0: log_mase.append(math.log(m / sn_m))
                        if sn_c > 0 and c > 0: log_crps.append(math.log(c / sn_c))
                except (ValueError, IndexError):
                    continue
    except Exception:
        return None, None, 0
    mase = math.exp(sum(log_mase) / len(log_mase)) if log_mase else None
    crps = math.exp(sum(log_crps) / len(log_crps)) if log_crps else None
    return mase, crps, len(log_mase)
