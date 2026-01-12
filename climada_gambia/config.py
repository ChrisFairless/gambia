CONFIG = {
    "data_dir": "/Users/chrisfairless/Data/UNU/gambia2025/inputs",
    "output_dir": "/Users/chrisfairless/Data/UNU/gambia2025/outputs",
    "default_analysis_name": "uncalibrated",
    "uncalibrated_analysis_name": "uncalibrated",  # Used for calculating uncalibrated exceedance curves
    "hazard": {
        # hazards are stored in the input data folder / hazard / f'{hazard_type}_{hazard_source}' / haz
        # but can be overridden by setting a "dir" value
        "flood": {
            "aqueduct": {
                "present": {
                    "scenario": "present",
                    "year": 1980,
                    "files": "GMB_inunriver_historical_WATCH_1980.hdf5",
                    "enabled": True
                },
                "RCP4.5_2050": {
                    "scenario": "RCP4.5",
                    "year": 2050,
                    "files": [
                        "GMB_inunriver_rcp4p5_MIROC-ESM-CHEM_2050.hdf5",
                        "GMB_inunriver_rcp4p5_ALL_2050.hdf5",
                        "GMB_inunriver_rcp4p5_NorESM1-M_2050.hdf5",
                        "GMB_inunriver_rcp4p5_GFDL-ESM2M_2050.hdf5",
                        "GMB_inunriver_rcp4p5_HadGEM2-ES_2050.hdf5",
                        "GMB_inunriver_rcp4p5_IPSL-CM5A-LR_2050.hdf5"
                    ],
                    "enabled": True
                },
                "RCP8.5_2050": {
                    "scenario": "RCP8.5",
                    "year": 2050,
                    "files": [
                        "GMB_inunriver_rcp8p5_IPSL-CM5A-LR_2050.hdf5",
                        "GMB_inunriver_rcp8p5_MIROC-ESM-CHEM_2050.hdf5",
                        "GMB_inunriver_rcp8p5_ALL_2050.hdf5",
                        "GMB_inunriver_rcp8p5_NorESM1-M_2050.hdf5",
                        "GMB_inunriver_rcp8p5_GFDL-ESM2M_2050.hdf5",
                        "GMB_inunriver_rcp8p5_HadGEM2-ES_2050.hdf5"
                    ],
                    "enabled": True
                }
            },
            "jrc": {
                "present": {
                    "files": "",
                    "enabled": False
                }
            }
        }
    },
    "exposures": {
        # exposures are stored in the input data folder / exposures / f'{exposure_type}_{exposure_source}' / exp
        # but can be overridden by setting a "dir" value
        "population": {
            "GHS": {
                "present": {
                    "files": "ghs_pop_GMB.hdf5",
                    "enabled": True
                }
            }
        },
        "housing": {
            "BEM": {
                "present": {
                    "files": "gmb_bem_1x1_valfis.csv",
                    "enabled": False
                }
            },
            "GHS": {
                "present": {
                    "files": "ghs_hh_GMB.hdf5",
                    "enabled": True
                }
            }
        },
        "livestock": {
            "GLW4": {
                "present": {
                    "files": [
                        "glw4_buffalo_GMB_5as.hdf5",
                        "glw4_cattle_GMB_5as.hdf5",
                        "glw4_goats_GMB_5as.hdf5",
                        "glw4_pigs_GMB_5as.hdf5",
                        "glw4_sheep_GMB_5as.hdf5"
                    ],
                    "enabled": True
                }
            }
        },
        "agriculture": {
            "IUCN": {
                "present": {
                    "files": "iucn_agriculture_GMB.hdf5",
                    "enabled": True
                }
            }
        },
        "energy": {
            "NCCS": {
                "present": {
                    "files": "energy_nccs_downscaled_GMB.hdf5",
                    "enabled": True
                }
            }
        },
        "manufacturing": {
            "NCCS": {
                "present": {
                    "files": "manufacturing_nccs_downscaled_GMB.hdf5",
                    "enabled": True
                }
            }
        },
        "services": {
            "NCCS": {
                "present": {
                    "files": "services_litpop_GMB.h5",
                    "enabled": True
                }
            }
        },
        "roads": {
            "OSM": {
                "present": {
                    "dir": "exposures/roads_OSM/",
                    "enabled": False
                }
            }
        },
        "economic_assets": {
            "litpop": {
                "present": {
                    "files": "economic_assets_litpop_GMB.h5",
                    "enabled": True
                }
            }
        }
    },

    "impact_functions": [
        {
            "hazard_type": "flood",
            "hazard_source": "aqueduct",
            "impfs": [

                # Uncalibrated
                {
                    "exposure_type": "population",
                    "exposure_source": "GHS",
                    "impact_type": "displaced",
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_housing_uncalibrated.csv",
                    "thresholds": {
                        'affected': 0.1
                    },
                    "enabled": True
                },
                {
                    "exposure_type": "housing",
                    "exposure_source": "GHS",
                    "impact_type": "economic_loss",
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_housing_uncalibrated.csv",
                    "thresholds": {
                        'affected': 0.1,
                        'damaged': 0.5,
                        'destroyed': 0.8
                    },
                    "scale_impf": 2,
                    "enabled": True
                },
                {
                    "exposure_type": "housing",
                    "exposure_source": "BEM",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_housing_uncalibrated.csv",
                    "thresholds": {
                        'affected': 0.1,
                        'damaged': 0.5,
                        'destroyed': 0.8
                    },
                    "enabled": False
                },
                {
                    "exposure_type": "agriculture",
                    "exposure_source": "IUCN",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_agriculture_uncalibrated.csv",
                    "thresholds": {
                        'affected': 0.1,
                        'damaged': 0.5
                    },
                    "scale_impf": 1.5,
                    "enabled": True
                },
                # {
                #     "exposure_type": "livestock",
                #     "exposure_source": "GLW4",
                #     "impact_type": "affected",
                #     "calibrated": False,
                #     "dir": "impact_functions/uncalibrated/",
                #     "files": "impf_river_flood_livestock_uncalibrated.csv",
                #     "enabled": True
                # },
                {
                    "exposure_type": "livestock",
                    "exposure_source": "GLW4",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_livestock_uncalibrated.csv",
                    "thresholds": {
                        'affected': 0.1,
                        'damaged': 0.5
                    },
                    "scale_impf": 0.5,
                    "enabled": True
                },
                {
                    "exposure_type": "manufacturing",
                    "exposure_source": "NCCS",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_manufacturing_uncalibrated.csv",
                    "scale_impf": 2,
                    "enabled": True
                },
                {
                    "exposure_type": "energy",
                    "exposure_source": "NCCS",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_energy_uncalibrated.csv",
                    "scale_impf": 2,
                    "enabled": True
                },
                {
                    "exposure_type": "services",
                    "exposure_source": "NCCS",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_services_uncalibrated.csv",
                    "scale_impf": 2,
                    "enabled": True
                },
                {
                    "exposure_type": "roads",
                    "exposure_source": "OSM",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_services_uncalibrated.csv",
                    "enabled": False
                },
                {
                    "exposure_type": "economic_assets",
                    "exposure_source": "litpop",
                    "impact_type": "economic_loss",
                    "calibrated": False,
                    "dir": "impact_functions/uncalibrated/",
                    "files": "impf_river_flood_housing_uncalibrated.csv",
                    "scale_impf": 2,
                    "enabled": True
                },
            ]
        }
    ]
}