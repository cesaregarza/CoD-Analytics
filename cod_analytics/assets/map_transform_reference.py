import numpy as np
from matplotlib import pyplot as plt

from cod_analytics.assets import asset_path
from cod_analytics.assets.map_images import MapPathRemap
from cod_analytics.classes import TransformReference
from cod_analytics.math.homography import Homography, HomographyCorrection


class MapCalibrationReference:
    mp_backlot2 = [
        [[-788.4292, 2300.2275], [171, 642]],
        [[264.85834, 2508.9119], [141, 506]],
        [[-945.97455, 1701.4312], [249, 641]],
        [[1336.8588, 1702.8577], [252, 352]],
        [[659.02216, -1248.1263], [651, 440]],
        [[-277.72818, -1395.3425], [672, 573]],
    ]
    mp_m_speed = [
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
    ]
    mp_rust = [
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
    ]
    mp_hardhat = [
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
    ]
    mp_aniyah_tac = [
        [[644.858, -804.492], [291, 343]],
        [[5463.13, -576.265], [810, 371]],
        [[4934.44, 689.206], [754, 507]],
        [[4631.15, 985.654], [724, 541]],
        [[678.175, 1237.88], [293, 564]],
        [[3359.51, -1104.26], [581, 310]],
        [[2445.95, -679.125], [485, 357]],
    ]
    mp_broadcast2 = [
        [[12870.9, 19413.5], [380, 636]],
        [[14393.13, 16880.99], [572, 313]],
        [[14339.98, 17132.11], [568, 346]],
        [[12971.96, 17054.63], [383, 323]],
        [[13774.26, 18193.3], [451, 639]],
        [[13510.13, 18740.86], [354, 323]],
        [[14638.03, 1811.93], [580, 642]],
        [[11692.1, 18815.45], [231, 562]],
    ]

    @staticmethod
    def get_transform(map_id: str) -> Homography:
        if not hasattr(MapCalibrationReference, map_id):
            raise ValueError(f"Map {map_id} not found")
        points = np.array(getattr(MapCalibrationReference, map_id))
        source_points = points[:, 0, :]
        target_points = points[:, 1, :]
        target_points[:, 1] = 1024 - target_points[:, 1]
        hom = Homography()
        hom.fit(source_points, target_points)
        return hom


class MapSourceOfTruthPoints:
    mp_aniyah_tac = TransformReference(  # Scale seems wrong
        map_left=-1275, map_right=6725, map_top=-3215, map_bottom=4785
    )
    mp_backlot2 = TransformReference(  # Scale is definitely wrong
        map_left=-2488,
        map_right=3212,
        map_top=-2862,
        map_bottom=2838,
    )
    mp_broadcast2 = TransformReference(  # Scale seems right, translated greatly
        map_left=-6329, map_right=3714, map_top=-2130, map_bottom=7913
    )
    mp_cave_am = TransformReference(
        map_left=-3760, map_right=5100, map_top=-3670, map_bottom=5190
    )
    mp_crash2 = TransformReference(
        map_left=-2655,
        map_right=3335,
        map_top=-2845,
        map_bottom=3145,
    )
    mp_deadzone = TransformReference(
        map_left=-6130,
        map_right=6140,
        map_top=-6152,
        map_bottom=6118,
    )
    mp_emporium = TransformReference(  # Scale is wrong
        map_left=-3765,
        map_right=4005,
        map_top=-4055,
        map_bottom=3715,
    )
    mp_garden = TransformReference(  # Needs minor scaling.
        map_left=-4080,
        map_right=3820,
        map_top=-4220,
        map_bottom=3680,
        map_rotation=180.0,
    )
    mp_hackney_am = TransformReference(
        map_left=-2780,
        map_right=3540,
        map_top=-3310,
        map_bottom=3010,
    )
    mp_harbor = TransformReference(
        map_left=-2230, map_right=6130, map_top=-5120, map_bottom=3240
    )
    mp_hardhat = TransformReference(
        map_left=-1375, map_right=2765, map_top=-2090, map_bottom=2050
    )
    mp_hideout = TransformReference(
        map_left=-2670, map_right=2730, map_top=-3060, map_bottom=2340
    )
    mp_killhouse = TransformReference(
        map_left=-2929,
        map_right=2992,
        map_top=-2984,
        map_bottom=2937,
        map_rotation=90.0,
    )
    mp_m_speed = TransformReference(
        map_left=-3560,
        map_right=2010,
        map_top=-1000,
        map_bottom=4570,
        map_rotation=90.0,
    )
    mp_malyshev = TransformReference(
        map_left=-6329, map_right=3714, map_top=-2130, map_bottom=7913
    )
    mp_oilrig = TransformReference(
        map_left=-3800,
        map_right=3800,
        map_top=-4110,
        map_bottom=3490,
        map_rotation=90.0,
    )
    mp_petrograd = TransformReference(
        map_left=-4090, map_right=4600, map_top=-4091, map_bottom=4599
    )
    mp_piccadilly = TransformReference(
        map_left=-4300, map_right=4040, map_top=-5320, map_bottom=3020
    )
    mp_runner = TransformReference(
        map_left=-4109, map_right=4101, map_top=-4120, map_bottom=4090
    )
    mp_rust = TransformReference(
        map_left=-815, map_right=1955, map_top=-605, map_bottom=2165
    )
    mp_scrapyard = TransformReference(
        map_left=-28144,
        map_right=-23514,
        map_top=-13022,
        map_bottom=-8392,
        map_rotation=105.0,
    )
    mp_shipment = TransformReference(
        map_left=-2000, map_right=1360, map_top=303, map_bottom=3663
    )
    mp_spear = TransformReference(
        map_left=-3823, map_right=4347, map_top=-3600, map_bottom=4570
    )
    mp_vacant = TransformReference(
        map_left=744, map_right=5944, map_top=-1000, map_bottom=4200
    )
    mp_village2 = TransformReference(
        map_left=-3641, map_right=3769, map_top=-3655, map_bottom=3755
    )

    @staticmethod
    def get(map_id: str) -> TransformReference:
        if not hasattr(MapSourceOfTruthPoints, map_id):
            raise ValueError(f"Map {map_id} not found")
        return getattr(MapSourceOfTruthPoints, map_id)


class MapCorrections:
    mp_aniyah_tac = HomographyCorrection(
        scale=(0.845, 0.8512), translate=(2.0, -10.0), center=(512, 512)
    )
    mp_backlot2 = HomographyCorrection(
        scale=0.77,
        center=(512, 512),
        translate=(-20, -38),
        rotate=270.0,
        rad=False,
    )
    mp_broadcast2 = HomographyCorrection(
        scale=1.243,
        center=(512, 512),
        translate=(-1921, 1963),
    )
    mp_emporium = HomographyCorrection(
        scale=(0.79, 0.81),
        center=(512, 512),
        translate=(8, -15),
        rotate=90.0,
        rad=False,
    )
    mp_garden = HomographyCorrection(
        scale=1.03,
        center=(512, 512),
        translate=(-3, 0),
        rotate=270.0,
        rad=False,
    )
    mp_hackney_am = HomographyCorrection(
        center=(512, 512),
        rotate=270.0,
        rad=False,
    )
    mp_harbor = HomographyCorrection(
        center=(512, 512),
        translate=(0, -26),
        rotate=270.0,
        rad=False,
    )
    mp_hardhat = HomographyCorrection(
        center=(512, 512),
        translate=(-26, -30),
    )
    mp_hideout = HomographyCorrection(
        scale=0.725,
        center=(512, 512),
        translate=(0, 5),
        rotate=90.0,
        rad=False,
    )
    mp_killhouse = HomographyCorrection(
        center=(512, 512),
        rotate=90.0,
        rad=False,
    )
    mp_malyshev = HomographyCorrection(
        center=(512, 512),
        rotate=270.0,
        rad=False,
    )
    mp_oilrig = HomographyCorrection(
        center=(512, 512),
        translate=(-3, 3),
        rotate=90.0,
        rad=False,
    )
    mp_petrograd = HomographyCorrection(
        center=(512, 512),
        rotate=270.0,
        rad=False,
    )
    mp_scrapyard = HomographyCorrection(
        scale=0.71,
        center=(512, 512),
        translate=(-5, -8),
        rotate=120.0,
        rad=False,
    )
    mp_spear = HomographyCorrection(
        center=(512, 512),
        rotate=270.0,
        rad=False,
    )
    mp_vacant = HomographyCorrection(
        scale=0.83,
        center=(512, 512),
        translate=(35, 15),
        rotate=315.0,
        rad=False,
    )
    mp_village2 = HomographyCorrection(
        scale=0.83,
        center=(512, 512),
        translate=(10, -10),
        rotate=270.0,
        rad=False,
    )

    @staticmethod
    def get(map_id: str) -> HomographyCorrection:
        if not hasattr(MapCorrections, map_id):
            return HomographyCorrection()
        return getattr(MapCorrections, map_id)


def retrieve_minimap_image(map_id: str) -> np.ndarray:
    """Retrieve the minimap image for a given map_id as a numpy array.

    Args:
        map_id (str): The map_id to retrieve the minimap for

    Returns:
        np.ndarray: The minimap image as a numpy array
    """

    image_path = asset_path / "map_images" / MapPathRemap.remap(map_id)
    return plt.imread(image_path)