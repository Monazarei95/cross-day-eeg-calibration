const studyScores = {
  "participants": [
    {
      "participant": 1,
      "method": "classical",
      "raw": 0.5770833333333333,
      "ocular": 0.61875,
      "random": 0.5824652777777777,
      "pca": 0.6215277777777779
    },
    {
      "participant": 1,
      "method": "frozen_ssl",
      "raw": 0.4624999999999999,
      "ocular": 0.5722222222222222,
      "random": 0.470703125,
      "pca": 0.55625
    },
    {
      "participant": 2,
      "method": "classical",
      "raw": 0.33958333333333335,
      "ocular": 0.42222222222222217,
      "random": 0.34171006944444443,
      "pca": 0.4305555555555555
    },
    {
      "participant": 2,
      "method": "frozen_ssl",
      "raw": 0.2694444444444445,
      "ocular": 0.33194444444444443,
      "random": 0.26588541666666665,
      "pca": 0.3298611111111111
    },
    {
      "participant": 3,
      "method": "classical",
      "raw": 0.5868055555555555,
      "ocular": 0.6430555555555556,
      "random": 0.5918836805555555,
      "pca": 0.6659722222222222
    },
    {
      "participant": 3,
      "method": "frozen_ssl",
      "raw": 0.4840277777777778,
      "ocular": 0.5673611111111112,
      "random": 0.4859375,
      "pca": 0.5201388888888888
    },
    {
      "participant": 4,
      "method": "classical",
      "raw": 0.37152777777777773,
      "ocular": 0.36944444444444435,
      "random": 0.37348090277777773,
      "pca": 0.3895833333333333
    },
    {
      "participant": 4,
      "method": "frozen_ssl",
      "raw": 0.3618055555555556,
      "ocular": 0.3541666666666667,
      "random": 0.3549479166666667,
      "pca": 0.3520833333333333
    },
    {
      "participant": 5,
      "method": "classical",
      "raw": 0.3076388888888889,
      "ocular": 0.31111111111111117,
      "random": 0.3056857638888889,
      "pca": 0.2979166666666667
    },
    {
      "participant": 5,
      "method": "frozen_ssl",
      "raw": 0.2548611111111111,
      "ocular": 0.26736111111111105,
      "random": 0.257421875,
      "pca": 0.28541666666666665
    },
    {
      "participant": 6,
      "method": "classical",
      "raw": 0.32152777777777775,
      "ocular": 0.3423611111111111,
      "random": 0.3282552083333333,
      "pca": 0.3347222222222222
    },
    {
      "participant": 6,
      "method": "frozen_ssl",
      "raw": 0.3104166666666667,
      "ocular": 0.31180555555555556,
      "random": 0.3125434027777778,
      "pca": 0.30069444444444443
    },
    {
      "participant": 7,
      "method": "classical",
      "raw": 0.48263888888888895,
      "ocular": 0.5145833333333334,
      "random": 0.5014756944444443,
      "pca": 0.5055555555555555
    },
    {
      "participant": 7,
      "method": "frozen_ssl",
      "raw": 0.3479166666666666,
      "ocular": 0.40902777777777777,
      "random": 0.3676215277777778,
      "pca": 0.37569444444444444
    },
    {
      "participant": 8,
      "method": "classical",
      "raw": 0.5187499999999999,
      "ocular": 0.5180555555555555,
      "random": 0.5166232638888888,
      "pca": 0.5347222222222222
    },
    {
      "participant": 8,
      "method": "frozen_ssl",
      "raw": 0.42986111111111114,
      "ocular": 0.45902777777777776,
      "random": 0.4242621527777778,
      "pca": 0.45138888888888895
    },
    {
      "participant": 9,
      "method": "classical",
      "raw": 0.43611111111111117,
      "ocular": 0.46736111111111106,
      "random": 0.4311197916666667,
      "pca": 0.4729166666666666
    },
    {
      "participant": 9,
      "method": "frozen_ssl",
      "raw": 0.4131944444444445,
      "ocular": 0.44305555555555554,
      "random": 0.41332465277777775,
      "pca": 0.4354166666666666
    }
  ],
  "summary": {
    "frozen_ssl": {
      "mean_accuracy": {
        "raw": 0.37044753086419757,
        "ocular": 0.4128858024691358,
        "random": 0.3725163966049383,
        "pca": 0.4007716049382716
      },
      "paired_differences": {
        "ocular_minus_raw": {
          "mean": 0.042438271604938266,
          "ci95": [
            0.012207428140933521,
            0.07266911506894301
          ],
          "improved": 8
        },
        "ocular_minus_random": {
          "mean": 0.040369405864197516,
          "ci95": [
            0.012469399818309596,
            0.06826941191008544
          ],
          "improved": 7
        },
        "ocular_minus_pca": {
          "mean": 0.012114197530864203,
          "ci95": [
            -0.002394661019054783,
            0.026623056080783192
          ],
          "improved": 8
        }
      }
    },
    "classical": {
      "mean_accuracy": {
        "raw": 0.437962962962963,
        "ocular": 0.4674382716049382,
        "random": 0.4414110725308642,
        "pca": 0.47260802469135804
      },
      "paired_differences": {
        "ocular_minus_raw": {
          "mean": 0.029475308641975315,
          "ci95": [
            0.007805324240786754,
            0.05114529304316388
          ],
          "improved": 7
        },
        "ocular_minus_random": {
          "mean": 0.026027199074074112,
          "ci95": [
            0.004884266018007264,
            0.04717013213014096
          ],
          "improved": 8
        },
        "ocular_minus_pca": {
          "mean": -0.005169753086419753,
          "ci95": [
            -0.015301750554573577,
            0.00496224438173407
          ],
          "improved": 3
        }
      }
    }
  }
};
