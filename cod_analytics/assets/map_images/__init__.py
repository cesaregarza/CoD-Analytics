from cod_analytics.classes import RemapClass


class MapRemap(RemapClass):
    mp_deadzone = "Arklov Peak"
    mp_aniyah_tac = "Aniyah Incursion"
    mp_hackney_am = "Hackey Yard"
    mp_backlot2 = "Talsik Backlot"
    mp_runner = "Gun Runner"
    mp_scrapyard = "Scrapyard"
    mp_cave_am = "Azhir Cave"
    mp_crash2 = "Crash"
    mp_oilrig = "Petrov Oil Rig"
    mp_hardhat = "Hardhat"
    mp_garden = "Cheshire Park"
    mp_rust = "Rust"
    mp_spear = "Rammaza"
    mp_hideout = "Khandor Hideout"
    mp_petrograd = "St. Petrograd"
    mp_shipment = "Shipment"
    mp_killhouse = "Killhouse"
    mp_m_speed = "Shoothouse"
    mp_harbor = "Suldal Harbor"
    mp_emporium = "Atlas Superstore"
    mp_vacant = "Vacant"
    mp_village2 = "Hovec Sawmill"
    mp_broadcast2 = "Broadcast"
    mp_piccadilly = "Picadilly"
    mp_malyshev = "Tank Factory"


class MapPathRemap(RemapClass):
    mp_deadzone = "ArklovPeak.png"
    mp_aniyah_tac = "AniyahIncursion.png"
    mp_hackney_am = "HackneyYard.png"
    mp_backlot2 = "TalsikBacklot.png"
    mp_runner = "GunRunner.png"
    mp_scrapyard = "Scrapyard.png"
    mp_cave_am = "AzhirCave.png"
    mp_crash2 = "Crash.png"
    mp_oilrig = "PetrovOilRig.png"
    mp_hardhat = "Hardhat.png"
    mp_garden = "CheshirePark.png"
    mp_rust = "Rust.png"
    mp_spear = "Rammaza.png"
    mp_hideout = "KhandorHideout.png"
    mp_petrograd = "StPetrograd.png"
    mp_shipment = "Shipment.png"
    mp_killhouse = "Killhouse.png"
    mp_m_speed = "Shoothouse.png"
    mp_harbor = "SuldalHarbor.png"
    mp_emporium = "AtlasSuperstore.png"
    mp_vacant = "Vacant.png"
    mp_village2 = "HovecSawmill.png"
    mp_broadcast2 = "Broadcast.png"
    mp_piccadilly = "Picadilly.png"
    mp_malyshev = "TankFactory.png"


class MapRemapSimple(RemapClass):
    arklov = "mp_deadzone"
    aniyah = "mp_aniyah_tac"
    hackney = "mp_hackey_am"
    backlot = "mp_backlot2"
    gun_runner = "mp_runner"
    scrapyard = "mp_scrapyard"
    cave = "mp_cave_am"
    crash = "mp_crash2"
    oilrig = "mp_oilrig"
    hardhat = "mp_hardhat"
    cheshire_park = "mp_garden"
    rust = "mp_rust"
    rammaza = "mp_rust"
    khandor = "mp_hideout"
    petrograd = "mp_petrograd"
    shipment = "mp_shipment"
    killhouse = "mp_killhouse"
    shoothouse = "mp_shoothouse"
    harbor = "mp_harbor"
    superstore = "mp_emporium"
    vacant = "mp_vacant"
    sawmill = "mp_village2"
    broadcast = "mp_broadcast2"
    piccadilly = "mp_piccadilly"
    tank_factory = "mp_malyshev"


map_calibration_points = {
    "mp_backlot2": [
        [[-788.4292, 2300.2275], [171, 642]],
        [[264.85834, 2508.9119], [141, 506]],
        [[-945.97455, 1701.4312], [249, 641]],
        [[1336.8588, 1702.8577], [252, 352]],
        [[659.02216, -1248.1263], [651, 440]],
        [[-277.72818, -1395.3425], [672, 573]],
    ],
    "mp_m_speed": [
        [[-763.9451, 688.8744], [713, 509]],
        [[-1632.9628, 2285.7134], [440, 664]],
        [[-555.8494, 2200.0884], [435, 471]],
        [[-846.9074, 260.04932], [793, 523]],
        [[-1525.6207, 1989.349], [476, 648]],
        [[-1538.0742, 3118.4974], [265, 655]],
        [[-991.1258, 1486.244], [566, 550]],
        [[-122.53425, 2862.4595], [319, 394]],
        [[-1061.43, 80.625], [826, 566]],
        [[-1104.04, 1648.633], [541, 570]],
        [[215.3582, 2512.055], [382, 331]],
        [[-752.875, 735.1619], [709, 510]],
    ],
    "mp_rust": [
        [[-338.944950, 718.45306], [192, 511]],
        [[-376.326300, 1201.035000], [162, 346]],
        [[486.047060, 991.72340], [477, 423]],
        [[1615.365700, 88.119580], [841, 712]],
        [[1616.267800, 580.829400], [867, 559]],
        [[1426.600300, 1544.32020], [781, 218]],
        [[1105.741700, -193.39981], [691, 852]],
        [[-258.584720, 352.558800], [201, 647]],
        [[582.99940, -151.51290], [498, 845]],
        [[100.633360, 1699.62260], [335, 176]],
        [[1098.810800, 1327.13850], [694, 280]],
        [[686.72940, 950.459000], [540, 437]],
        [[-431.154750, -28.447388], [149, 793]],
    ],
    "mp_hardhat": [
        [[1317.54, -383.261], [608, 464]],
        [[1815, 1426], [706, 823]],
        [[667.545, 872.165], [476, 713]],
        [[212.640, 1236.56], [389, 784]],
        [[1775.14, -583.573], [691, 434]],
        [[1079.21, -953.671], [561, 358]],
        [[161.108, -1017.13], [382, 347]],
        [[1055.1, -1061.15], [142, 331]],
        [[1533.83, 159.013], [642, 567]],
        [[-1052.55, -840.432], [142, 379]],
    ],
    "mp_aniyah_tac": [
        [[644.858, -804.492], [291, 343]],
        [[5463.13, -576.265], [810, 371]],
        [[4934.44, 689.206], [754, 507]],
        [[4631.15, 985.654], [724, 541]],
        [[678.175, 1237.88], [293, 564]],
        [[3359.51, -1104.26], [581, 310]],
        [[2445.95, -679.125], [485, 357]],
    ],
    "mp_broadcast2": [
        [[12870.9, 19413.5], [380, 636]],
        [[14393.13, 16880.99], [572, 313]],
        [[14339.98, 17132.11], [568, 346]],
        [[12971.96, 17054.63], [383, 323]],
        [[13774.26, 18193.3], [451, 639]],
        [[13510.13, 18740.86], [354, 323]],
        [[14638.03, 1811.93], [580, 642]],
        [[11692.1, 18815.45], [231, 562]],
    ],
}

source_of_truth_points = {
    "mp_aniyah_tac": {
        "map_reference_points": (-1275, 6725, -3215, 4785),
        "image_reference_points": (1024, 1024),
    },
    "mp_backlot2": {
        "map_reference_points": (-2488, 3212, -2862, 2838),
        "image_reference_points": (1000, 1000),
    },
    "mp_broadcast2": {
        "map_reference_points": (-6329, 3714, -2130, 7913),
        "image_reference_points": (1022, 1022),
    },
    "mp_cave_am": {
        "map_reference_points": (-3760, 5100, -3670, 5190),
        "image_reference_points": (1024, 1024),
    },
    "mp_crash2": {
        "map_reference_points": (-2655, 3335, -2845, 3145),
        "image_reference_points": (753, 753),
    },
    "mp_deadzone": {
        "map_reference_points": (-6130, 6140, -6152, 6118),
        "image_reference_points": (1024, 1024),
    },
    "mp_emporium": {
        "map_reference_points": (-3765, 4005, -4055, 3715),
        "image_reference_points": (1024, 1024),
    },
    "mp_garden": {
        "map_reference_points": (-4080, 3820, -4220, 3680),
        "image_reference_points": (1024, 1024),
    },
    "mp_hackney_am": {
        "map_reference_points": (-2780, 3540, -3310, 3010),
        "image_reference_points": (1024, 1024),
    },
    "mp_harbor": {
        "map_reference_points": (-2230, 6130, -5120, 3240),
        "image_reference_points": (1000, 1000),
    },
    "mp_hardhat": {
        "map_reference_points": (-1375, 2765, -2090, 2050),
        "image_reference_points": (1022, 1022),
    },
    "mp_hideout": {
        "map_reference_points": (-2670, 2730, -3060, 2340),
        "image_reference_points": (1024, 1024),
    },
    "mp_killhouse": {
        "map_reference_points": (-2929, 2992, -2984, 2937),
        "image_reference_points": (1024, 1024),
    },
    "mp_m_speed": {
        "map_reference_points": (-3560, 2010, -1000, 4570),
        "image_reference_points": (1024, 1024),
    },
    "mp_malyshev": {
        "map_reference_points": (-6329, 3714, -2130, 7913),
        "image_reference_points": (1022, 1022),
    },
    "mp_oilrig": {
        "map_reference_points": (-3800, 3800, -4110, 3490),
        "image_reference_points": (1022, 1022),
    },
    "mp_petrograd": {
        "map_reference_points": (-4090, 4600, -4091, 4599),
        "image_reference_points": (1024, 1024),
    },
    "mp_piccadilly": {
        "map_reference_points": (-4300, 4040, -5320, 3020),
        "image_reference_points": (1024, 1024),
    },
    "mp_runner": {
        "map_reference_points": (-4109, 4101, -4120, 4090),
        "image_reference_points": (1024, 1024),
    },
    "mp_rust": {
        "map_reference_points": (-815, 1955, -605, 2165),
        "image_reference_points": (1000, 1000),
    },
    "mp_scrapyard": {
        "map_reference_points": (-28144, -23514, -13022, -8392),
        "image_reference_points": (1024, 1024),
    },
    "mp_shipment": {
        "map_reference_points": (-2000, 1360, 303, 3663),
        "image_reference_points": (1024, 1024),
    },
    "mp_spear": {
        "map_reference_points": (-3823, 4347, -3600, 4570),
        "image_reference_points": (1024, 1024),
    },
    "mp_vacant": {
        "map_reference_points": (744, 5944, -1000, 4200),
        "image_reference_points": (1024, 1024),
    },
    "mp_village2": {
        "map_reference_points": (-3641, 3769, -3655, 3755),
        "image_reference_points": (1024, 1024),
    },
}
