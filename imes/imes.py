from math import sqrt, pi

import numpy as np
from scipy.interpolate import interp1d


CUT_SPEEDS = np.array([0, 30, 60, 90])
WEAR_COEFFS = np.array([0, 42, 62, 78])
lerp_cutspeed_coeffs = interp1d(CUT_SPEEDS, WEAR_COEFFS)


def feed_rate(vc, feed_per_tooth):
    tool_diameter = 3.175
    num_teeth = 4
    rpm = vc / (pi * tool_diameter / 1000)
    return rpm * num_teeth * feed_per_tooth


def wdot(L, vc, feed_per_tooth):
    global lerp_cutspeed_coeffs

    alpha = lerp_cutspeed_coeffs(vc)
    fr = feed_rate(vc, feed_per_tooth) / 1000
    vbdot = alpha * fr / (2 * sqrt(L))
    return np.array([ fr, vbdot ])


def thermal_poly_a(h):
    return -4.581e-1 * h ** 5 + \
        1.813e1 * h ** 4 + \
        -2.518e2 * h ** 3 + \
        1.397e3 * h ** 2 + \
        -1.836e3 * h + \
        8.938e2


def thermal_poly_b(h):
    return 7.138e-5 * h ** 5 + \
        -2.715e-3 * h ** 4 + \
        3.578e-2 * h ** 3 + \
        -1.819e-1 * h ** 2 + \
        1.906e-1 * h + \
        -7.014e-1


def fz_to_h(fz):
    return 154.91 * fz


def thermal_damage_limit(wear, fz):
    h = fz_to_h(fz)
    return thermal_poly_a(h) * wear ** thermal_poly_b(h)


def feed_rate_damage_limit(wear):
    return 5.38e-4 * wear + 4e-3


def calc_profit(stats, **kwargs):
    machining_dist = kwargs.get("machining_dist", 400)
    tool_cost = kwargs.get("tool_cost", 15)
    overhead_per_hour = kwargs.get("overhead_per_hour", 100)
    tool_change_time_per_part = kwargs.get("tool_change_time", 1)
    proceeds_per_part = kwargs.get("proceeds_per_part", 200)
    raw_material_cost = kwargs.get("raw_material_cost", 90)
    qc_time_per_part = kwargs.get("qc_time_per_part", 5)
    qc_cost_per_part = kwargs.get("qc_cost_per_part", 8)

    feed_rate = stats['fr_avg']
    tool_dist = stats['slide_distance']

    tool_life_time = tool_dist * 1000 / feed_rate

    parts_per_tool = tool_dist * 1000 / machining_dist
    tool_cost_per_part = 1 / parts_per_tool * tool_cost
    tool_change_time_per_part = 1 / parts_per_tool * tool_change_time_per_part
    total_process_time_per_part = tool_change_time_per_part + \
        machining_dist / feed_rate + qc_time_per_part

    overhead_per_part = total_process_time_per_part * overhead_per_hour / 60
    total_cost_per_part = tool_cost_per_part + overhead_per_part \
        + qc_cost_per_part + raw_material_cost
    cost_per_min = total_cost_per_part / total_process_time_per_part
    proceeds_per_min = proceeds_per_part / total_process_time_per_part

    profit_per_min = proceeds_per_min - cost_per_min
    return profit_per_min


def calc_energy(stats, **kwargs):
    process_energy_per_min = kwargs.get("process_energy_per_min", 8.3e-5)
    tooling_energy_per_tool = kwargs.get("tooling_energy_per_tool", 10.)
    machining_dist = kwargs.get("machining_dist", 400)
    tool_change_time_per_part = kwargs.get("tool_change_time", 1)
    qc_time_per_part = kwargs.get("qc_time_per_part", 5)
    workpiece_energy = kwargs.get("workpiece_energy", 150)
    scrap_rate = kwargs.get("scrap_rate", 0.01)

    feed_rate = stats['fr_avg']
    tool_dist = stats['slide_distance']

    parts_per_tool = tool_dist * 1000 / machining_dist
    tool_change_time_per_part = 1 / parts_per_tool * tool_change_time_per_part
    total_process_time_per_part = tool_change_time_per_part + \
        machining_dist / feed_rate + qc_time_per_part

    process_energy_per_part = total_process_time_per_part * process_energy_per_min
    tooling_energy_per_part = 1 / parts_per_tool * tooling_energy_per_tool
    scrap_energy_per_part = scrap_rate * workpiece_energy

    total_energy = process_energy_per_part + tooling_energy_per_part \
        + scrap_energy_per_part

    return total_energy
