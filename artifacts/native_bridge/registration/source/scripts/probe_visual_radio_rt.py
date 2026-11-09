"""Bounded real CPU ray tracing and images from the same warehouse geometry.

This checks simulator synchronization only. It is not a measured-channel,
visual-prediction, collision, networking-throughput or control benchmark.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/workspace/upstream/service-world-reference/coupled_raytracer/environments/aws_warehouse/sionna')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('/workspace/training/visual-radio/rt'))
    parser.add_argument('--dino', action='store_true', help='Encode the two actual RT images using cached pretrained DINOv3')
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'artifacts/visual_radio/rt_probe/receipt.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipt = {'completed': False, 'status': 'running', 'stage': 'imports',
               'started_at_unix': started, 'simulation_only': True,
               'scope': 'Real full-warehouse CPU ray-tracing/render synchronization feasibility only',
               'versions': {name: importlib.metadata.version(name)
                            for name in ('sionna', 'sionna-rt', 'mitsuba', 'drjit')},
               'script_sha256': digest(Path(__file__)), 'poses': [],
               'source': {'repository': 'https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration',
                          'commit': '2d37c650bde5af6e5f69e899aa4e4b01243d3160', 'license': 'MIT',
                          'scene_path': str(ASSETS / 'warehouse.xml')},
               'claim_boundary': {'real_hardware_validation': False, 'trained_visual_predictor': False,
                                  'rf_material_calibrated_to_real_site': False, 'sota_established': False,
                                  'hidden_radio_labels_painted_into_images': False,
                                  'synthetic_channel_fallback': False}}

    def checkpoint(stage):
        receipt['stage'] = stage
        receipt['elapsed_seconds'] = time.time() - started
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print(stage, flush=True)

    try:
        import numpy as np
        import mitsuba as mi
        import drjit as dr
        from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, SceneObject, PathSolver
        from sionna.rt.utils.render import make_render_sensor

        receipt['mitsuba_variant'] = mi.variant()
        receipt['llvm_backend_available'] = bool(dr.has_backend(dr.JitBackend.LLVM))
        checkpoint('load_full_warehouse')
        scene = load_scene(str(ASSETS / 'warehouse.xml'), merge_shapes=False)
        scene.frequency = 5.8e9
        receipt['source']['files_sha256'] = {str(p.relative_to(ASSETS)): digest(p)
                                            for p in sorted(ASSETS.rglob('*')) if p.is_file()}
        receipt['original_scene_objects'] = len(scene.objects)
        receipt['world_bbox'] = str(scene.mi_scene.bbox())
        receipt['radio_materials'] = {name: {'relative_permittivity': float(np.asarray(mat.relative_permittivity).ravel()[0]),
                                            'conductivity_s_per_m': float(np.asarray(mat.conductivity).ravel()[0])}
                                      for name, mat in scene.radio_materials.items()}
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                    horizontal_spacing=.5, pattern='iso', polarization='V')
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                    horizontal_spacing=.5, pattern='iso', polarization='V')
        tx_positions = [[-1., -6., 1.8], [1., -6., 1.8]]
        rx_positions = [[1., -1., 1.], [-1., -1., 1.]]
        for i, position in enumerate(tx_positions):
            scene.add(Transmitter(f'tx_{i}', position=position))
        for i, position in enumerate(rx_positions):
            scene.add(Receiver(f'rx_{i}', position=position))
        receipt['radio_configuration'] = {'frequency_hz': 5.8e9, 'tx_positions_m': tx_positions,
                                           'rx_positions_m': rx_positions, 'antenna': 'Single isotropic V-polarized element',
                                           'desired_links': [[0, 0], [1, 1]], 'cross_interference_links': [[0, 1], [1, 0]],
                                           'coupling_index_order': '[receiver, transmitter]',
                                           'transmit_power': 'Unit-power channel coupling; no MAC/scheduling/noise/SINR model'}
        # An original, explicit rectangular metal obstacle is inserted into the
        # full warehouse; the identical mesh is used for optics and RF.
        vertices = [(-.25, -1., -1.5), (.25, -1., -1.5), (.25, 1., -1.5), (-.25, 1., -1.5),
                    (-.25, -1., 1.5), (.25, -1., 1.5), (.25, 1., 1.5), (-.25, 1., 1.5)]
        faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                 (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        blocker_file = args.work_dir / 'explicit_metal_blocker.ply'
        blocker_file.write_text('ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'
                                + '\n'.join(' '.join(map(str, v)) for v in vertices) + '\n'
                                + '\n'.join('3 ' + ' '.join(map(str, f)) for f in faces) + '\n')
        blocker = SceneObject(name='explicit_metal_blocker', fname=str(blocker_file),
                              radio_material=scene.radio_materials['itu_metal'], position=[0., -3., 1.5])
        scene.add(blocker)
        camera = Camera(position=[2., -7., 3.7], look_at=[0., -3., 1.3])
        width, height, fov = 256, 192, 60.
        focal = .5 * width / math.tan(math.radians(fov) / 2.)
        camera_to_world = camera.world_transform.matrix.numpy()[:, :, 0]
        sensor = make_render_sensor(scene, camera, (width, height), fov)
        local_rays = []
        for uv in ((.5, .5), (1., .5), (.5, 1.)):
            ray, _ = sensor.sample_ray(0., .5, mi.Point2f(uv), mi.Point2f(.5, .5))
            local_rays.append((camera_to_world[:3, :3].T @ np.asarray(ray.d).reshape(3)).tolist())
        assert abs(local_rays[0][0]) < 1e-6 and abs(local_rays[0][1]) < 1e-6
        assert local_rays[1][0] < 0. and local_rays[2][1] < 0.
        receipt['camera'] = {'position_world_m': [2., -7., 3.7], 'look_at_world_m': [0., -3., 1.3],
                              'resolution_wh': [width, height], 'horizontal_fov_degrees': fov,
                              'K_pixel_centers_at_integer_plus_half': [[focal, 0., width / 2.], [0., focal, height / 2.], [0., 0., 1.]],
                              'camera_to_world_mitsuba': camera_to_world.tolist(),
                              'camera_to_world_opencv': (camera_to_world @ np.diag([-1., -1., 1., 1.])).tolist(),
                              'sensor_local_rays_center_right_down': local_rays,
                              'convention': 'Mitsuba local +X left,+Y up,+Z forward; OpenCV +X right,+Y down,+Z forward. Raster origin upper left. Sensor center/right/down rays checked.',
                              'simulation_calibration': True, 'measured_camera_calibration': False}
        receipt['solver'] = {'max_depth': 2, 'samples_per_src': 2048, 'max_num_paths_per_src': 4096,
                              'los': True, 'specular_reflection': True, 'refraction': False,
                              'diffuse_reflection': False, 'diffraction': False, 'seed': 42,
                              'convergence_established': False, 'render_samples_per_pixel': 4}
        receipt['blocker'] = {'mesh_path': str(blocker_file), 'mesh_sha256': digest(blocker_file),
                               'dimensions_m': [.5, 2., 3.], 'material': 'itu_metal',
                               'positions_m': [[0., -3., 1.5], [0., 0., 1.5]],
                               'visual_color_semantics': 'Renderer material only; no channel/power/blockage labels overlaid'}
        earlier = output.parent / 'initial_geometry_attempt.json'
        if earlier.exists():
            receipt['placement_development'] = {
                'initial_receipt_path': str(earlier.relative_to(ROOT)), 'initial_receipt_sha256': digest(earlier),
                'reason': 'Initial devices across densely obstructed warehouse produced zero found paths at the bounded solver setting; not useful training evidence. Devices were repositioned into an aisle using geometry-only floor-ray checks. Both original blocker poses are unchanged.',
                'used_as_performance_comparison': False}
        solver = PathSolver()
        for index, position in enumerate(receipt['blocker']['positions_m']):
            blocker.position = position
            checkpoint(f'pose_{index}_ray_trace')
            paths = solver(scene, max_depth=2, samples_per_src=2048, max_num_paths_per_src=4096,
                           los=True, specular_reflection=True, diffuse_reflection=False,
                           refraction=False, diffraction=False, seed=42)
            a, tau = paths.cir(normalize_delays=False, out_type='numpy')
            # Single antenna, single time: [rx,rx_ant,tx,tx_ant,path,time].
            coefficients = a[:, 0, :, 0, :, 0]
            powers = np.sum(np.abs(coefficients) ** 2, axis=-1)
            coherent = np.abs(np.sum(coefficients, axis=-1)) ** 2
            array_path = args.work_dir / f'pose_{index}_channels.npz'
            np.savez_compressed(array_path, a=a, tau=tau, incoherent_path_power=powers,
                                coherent_center_frequency_power=coherent)
            checkpoint(f'pose_{index}_render_same_scene')
            image_path = args.work_dir / f'pose_{index}_rgb.png'
            scene.render_to_file(camera=camera, filename=str(image_path), fov=fov,
                                 resolution=(width, height), num_samples=4,
                                 show_devices=False, show_orientations=False)
            receipt['poses'].append({'pose_index': index, 'blocker_position_m': position,
                                      'cir_shape': list(a.shape), 'all_channel_coefficients_finite': bool(np.isfinite(a).all()),
                                      'incoherent_path_power_rx_tx': powers.tolist(),
                                      'coherent_center_frequency_power_rx_tx': coherent.tolist(),
                                      'valid_path_counts_rx_tx': np.asarray(paths.valid).sum(axis=-1).squeeze().tolist(),
                                      'channels_path': str(array_path), 'channels_sha256': digest(array_path),
                                      'rgb_path': str(image_path), 'rgb_sha256': digest(image_path),
                                      'same_scene_and_pose_for_rgb_and_channels': True,
                                      'power_interpretation': 'Incoherent sum |a_path|² and coherent |sum a_path|² are separately recorded; neither is throughput or observed interference traffic.'})
        values = [np.asarray(p['incoherent_path_power_rx_tx']) for p in receipt['poses']]
        validation = {'nonzero_desired_links_in_pose_1': bool(np.all(np.diag(values[1]) > 0)),
                       'nonzero_cross_links': bool(all(v[0, 1] > 0 and v[1, 0] > 0 for v in values)),
                       'coupling_changes_across_poses': bool(np.any(values[0] != values[1])),
                       'rendered_image_changes_across_poses': receipt['poses'][0]['rgb_sha256'] != receipt['poses'][1]['rgb_sha256'],
                       'same_geometry_used_for_radio_and_rgb': True,
                       'camera_projection_convention_checked': True}
        passed = all(validation.values())
        if args.dino:
            checkpoint('encode_actual_rt_images_with_pretrained_dinov3')
            import torch
            from PIL import Image
            from wirelessjev.visual_radio.dino import DinoFrame, FrozenDinoV3
            torch.set_num_threads(2)
            model = FrozenDinoV3.from_pretrained(local_files_only=True)
            frames = [DinoFrame(Image.open(p['rgb_path']).convert('RGB'), time.time(),
                                {'frame_id': f'rt-static-pose-{p["pose_index"]}', 'camera_id': 'calibrated-simulation-camera',
                                 'source_uri': p['rgb_path'], 'license': 'MIT upstream warehouse assets; original blocker geometry',
                                 'source_file_sha256': p['rgb_sha256']}, capture_time_seconds=None)
                      for p in receipt['poses']]
            encoded = model(frames)
            feature_path = args.work_dir / 'rt_dinov3_features.npz'
            np.savez_compressed(feature_path, patch_tokens=encoded.patch_tokens.cpu().numpy(),
                                patch_valid=encoded.patch_valid.cpu().numpy(),
                                patch_boxes_original_pixels=encoded.patch_boxes_original_pixels.cpu().numpy(),
                                capture_time_known=encoded.capture_time_known.cpu().numpy(),
                                observation_time_seconds=encoded.observation_time_seconds.cpu().numpy(),
                                cls_token=encoded.cls_token.cpu().numpy())
            receipt['dinov3'] = {'model': encoded.model_metadata, 'patch_tokens_shape': list(encoded.patch_tokens.shape),
                                  'feature_path': str(feature_path), 'feature_sha256': digest(feature_path),
                                  'image_geometry': list(encoded.image_geometry), 'provenance': list(encoded.provenance),
                                  'capture_time_known': encoded.capture_time_known.tolist(),
                                  'all_features_finite': bool(torch.isfinite(encoded.patch_tokens).all()),
                                  'between_pose_token_l2': float((encoded.patch_tokens[1] - encoded.patch_tokens[0]).norm()),
                                  'time_semantics': 'Independent static simulated poses; unknown capture time. Delivery field is probe ingestion only, not a timed robot stream.',
                                  'radio_labels_supplied_to_encoder': False, 'rf_predictor_trained': False}
        receipt.update(completed=True, validation=validation, geometry_radio_validation_passed=passed,
                       status='real_full_warehouse_rt_and_rgb_completed' if passed else 'completed_but_geometry_radio_validation_failed')
        checkpoint('complete')
    except Exception as error:
        receipt.update(status='failed_without_channel_fallback', error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        checkpoint(receipt['stage'])
        raise


if __name__ == '__main__':
    main()
