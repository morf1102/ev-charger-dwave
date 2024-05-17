# Placement of Charging Stations

Determining optimal locations to build new electric vehicle charging stations is a complex optimization problem.  Many factors should be taken into consideration, like existing charger locations, points of interest (POIs), quantity to build, etc. In this example, we take a look at how we might formulate this optimization problem and solve it using D-Wave's binary quadratic model (BQM) hybrid solver.

## Code Structure

```monospace
/.
│   .gitignore
│   main.ipynb
│   README.md
│   requirements.txt
│
├───data/
│       55-40.geojson
│       55.geojson
│       center.geojson
│       college.geojson
│       lightposts_curated.geojson
│
├───modules/
│   │   genetic.py
│   │   utils.py
│   └───__init__.py
│
└───readme_imgs/
        map.png
```

## Genetic Algorithm

This class implements a genetic algorithm to solve the problem of finding the optimal locations for new charging stations.

### Initialization

The `GeneticAlgorithm` class is initialized with the following parameters:

- `width` (int): Width of the grid.
- `height` (int): Height of the grid.
- `num_poi` (int): Number of points of interest.
- `num_cs` (int): Number of existing charging stations.
- `num_new_cs` (int): Number of new charging stations to place.
- `pop_size` (int, optional): Population size. Defaults to 100.
- `mutation_rate` (float, optional): Mutation rate. Defaults to 0.1.
- `crossover_rate` (float, optional): Crossover rate. Defaults to 0.5.
- `generations` (int, optional): Number of generations. Defaults to 500.
- `seed` (Optional[int], optional): Seed for random number generator. Defaults to None.

### Example

```python
from genetic import GeneticAlgorithm
from modules.utils import output_image 

ga = GeneticAlgorithm(width=10, height=10, num_poi=5, num_cs=3, num_new_cs=2)

ga.run()

output_image(ga.graph, ga.pois, ga.charging_stations, ga.best_solution)
```

## Utils

The `utils.py` file contains utility functions to support data loading, image output, and distance calculation.

### Functions

#### `load_data(folder_path: str) -> gpd.GeoDataFrame`

Load all geojson files in the specified folder and return a GeoDataFrame containing the data.

- **Arguments:**
  - `folder_path` (str): Path to the folder containing the geojson files.

- **Returns:**
  - `gpd.GeoDataFrame`: A GeoDataFrame containing all the data from the geojson files.

- **Example:**

  ```python
  from modules.utils import load_data

  folder_path = "./data"
  gdf = load_data(folder_path)
  ```
