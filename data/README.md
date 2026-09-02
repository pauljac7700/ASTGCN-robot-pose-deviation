# Data

The datasets used in this work are published separately, in
[pauljac7700/serial_robots_datasets](https://github.com/pauljac7700/serial_robots_datasets):
laser-tracker measurements of commanded against realised end-effector positions
for a Universal Robots UR5 and a Barrett WAM, in grid and random pose
distributions.

They are not mirrored here. Only the subset released in that repository is
public; other measurement sets recorded during this work remain with Tsinghua
University, and third-party datasets used for comparison belong to their
respective authors.

To reproduce the experiments, clone the dataset repository and point
`dataset_name` and `dataset_type` in the config files at the CSVs you want:

```bash
git clone https://github.com/pauljac7700/serial_robots_datasets.git
```

`data_acquisition/` holds the scripts that recorded the UR5 measurements, and
`adjacency_matrix.npy` the graph topology consumed by the models.
