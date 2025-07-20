import random
from pyqstem import PyQSTEM
from ase.visualize import view
import numpy as np


def simulate_tem_image(config, model):
    # -> extract STEM parameters from config

    config.TEM_image_kwargs.QSTEM_mode = "STEM"

    sim_stem = {'qstem': PyQSTEM(config.TEM_image_kwargs.QSTEM_mode),

                'atomic_model': model,

                'image_size': (config.TEM_image_kwargs.image_size, config.TEM_image_kwargs.image_size),

                'sampling': config.structure_kwargs.spatial_domain / config.TEM_image_kwargs.image_size,

                'probe': config.TEM_image_kwargs.STEM_kwargs.probe,
                # 'probe': 10,

                'slice_thickness': config.TEM_image_kwargs.STEM_kwargs.slice_thickness,

                'v0': random.uniform(config.TEM_image_kwargs.STEM_kwargs.v0[0],
                                     config.TEM_image_kwargs.STEM_kwargs.v0[1]),

                'alpha': np.random.uniform(config.TEM_image_kwargs.STEM_kwargs.alpha[0],
                                           config.TEM_image_kwargs.STEM_kwargs.alpha[1]),

                'defocus': random.uniform(config.TEM_image_kwargs.STEM_kwargs.defocus[0],
                                          config.TEM_image_kwargs.STEM_kwargs.defocus[1]),

                'Cs': random.uniform(config.TEM_image_kwargs.STEM_kwargs.Cs[0],
                                     config.TEM_image_kwargs.STEM_kwargs.Cs[1]),

                'astig_mag': random.uniform(config.TEM_image_kwargs.STEM_kwargs.astig_mag[0],
                                            config.TEM_image_kwargs.STEM_kwargs.astig_mag[1]),

                'astig_angle': random.uniform(config.TEM_image_kwargs.STEM_kwargs.astig_angle[0],
                                              config.TEM_image_kwargs.STEM_kwargs.astig_angle[1]),

                'aberrations': {'a33': config.TEM_image_kwargs.STEM_kwargs.a33,
                                'phi33': config.TEM_image_kwargs.STEM_kwargs.phi33},

                'add_local_norm': True,  # config.TEM_image_kwargs.add_local_norm,
                'add_noise': config.TEM_image_kwargs.add_noise,
                'noise_mean': config.TEM_image_kwargs.noise_mean,
                'noise_std': config.TEM_image_kwargs.noise_std,
                'noise_window_size': 1 * int(10 / config.TEM_image_kwargs.spot_size)}

    get_stem = SimSTEM(**sim_stem)

    stem_img = get_stem.get_img()
    sim_stem['Mean_pixel_intensity_STEM'] = np.mean(stem_img)

    # def simulate_hrtem_image(config, model):
    config.TEM_image_kwargs.QSTEM_mode = "TEM"

    sim_hrtem = {'qstem': PyQSTEM(config.TEM_image_kwargs.QSTEM_mode),

                 'atomic_model': model,

                 'image_size': (config.TEM_image_kwargs.image_size, config.TEM_image_kwargs.image_size),

                 'sampling': config.structure_kwargs.spatial_domain / config.TEM_image_kwargs.image_size,

                 'defocus': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.defocus[0],
                                           config.TEM_image_kwargs.HRTEM_kwargs.defocus[1]),

                 'Cs': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.Cs[0],
                                      config.TEM_image_kwargs.HRTEM_kwargs.Cs[1]),

                 'aberrations': {'a22': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.astig_mag[0],
                                                       config.TEM_image_kwargs.HRTEM_kwargs.astig_mag[1]),
                                 'phi22': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.astig_angle[0],
                                                         config.TEM_image_kwargs.HRTEM_kwargs.astig_angle[1])},

                 'focal_spread': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.focal_spread[0],
                                                config.TEM_image_kwargs.HRTEM_kwargs.focal_spread[1]),

                 'blur': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.blur[0],
                                        config.TEM_image_kwargs.HRTEM_kwargs.blur[1]),

                 'dose': random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.dose[0],
                                        config.TEM_image_kwargs.HRTEM_kwargs.dose[1]),

                 'MTF_param': [random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.MTF.c1[0],
                                              config.TEM_image_kwargs.HRTEM_kwargs.MTF.c1[1]),
                               random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.MTF.c2[0],
                                              config.TEM_image_kwargs.HRTEM_kwargs.MTF.c2[1]),
                               random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.MTF.c3[0],
                                              config.TEM_image_kwargs.HRTEM_kwargs.MTF.c3[1]),
                               random.uniform(config.TEM_image_kwargs.HRTEM_kwargs.MTF.c4[0],
                                              config.TEM_image_kwargs.HRTEM_kwargs.MTF.c4[1])],

                 'add_local_norm': config.TEM_image_kwargs.add_local_norm,
                 'add_noise': config.TEM_image_kwargs.add_noise,
                 'noise_mean': config.TEM_image_kwargs.noise_mean,
                 'noise_std': config.TEM_image_kwargs.noise_std,
                 'noise_window_size': 1 * int(10 / config.TEM_image_kwargs.spot_size)}

    get_hrtem = SimHRTEM(**sim_hrtem)
    hrtem_img = get_hrtem.get_img()
    sim_hrtem['Mean_pixel_intensity_HRTEM'] = np.mean(hrtem_img)

    # params_dict = {sim_stem, sim_hrtem}

    pops = ['qstem', 'atomic_model']

    for i in pops:
        sim_stem.pop(i)
        sim_hrtem.pop(i)

    params_dict = dict()
    params_dict['STEM Paramters'] = sim_stem
    params_dict['HRTEM Paramters'] = sim_hrtem

    # Clean up large objects
    del get_stem, get_hrtem

    # return stem_img, hrtem_img
    return stem_img, hrtem_img, params_dict
