# Day Trip Planner
If you plan a vacation but you want to have too many locations you want to visit, this script is for you. It chooses a limited number of locations from a list and groups them into 'day trips'. Day trips have a maximum length that determines how many locations can be visited. 'Favorite' locations are always included. The day trips are optimized using TSP. The script intends to minimize API calls.

The python script can be executed by Pyodide. If you don't need that, you can delete the async definitions and `.to_py()` function calls as well as make nearly all parameters global variables.

**What you need**

* Python / Pyodide
* [python-tsp](https://pypi.org/project/python_tsp/) library
* list of locations to visit (poi/stations); format explained in route.py
* list of locations to stay at (homes/centers); format explained in route.py
* an API that gets you the duration of a route from point A to point B (in minutes)
   * e.g. Google Maps Platform, Navitime for Japan (Google Maps does not offer transit times there)

## Algorithm

There are two algorithms: The slow algorithm just uses TSP and is very slow. The fast algorithm is a heuristic and is explained in the following.

### Determine routes to fetch

To minimize the API calls, not all routes from every point to every other are fetched. Instead, we connect the closest point to each location to form a route. In the second iteration, we connect the second closest point to the location. This is repeated, until there are at most that many connected subgraphs as the 'stay' attribute of the center/home dictates. Image: Example for stay=2, nearest neighbor = dark gray, second nearest = light gray

![Step1](https://github.com/user-attachments/assets/ab5aafee-d1d9-463e-9af7-f26a78b19f3b)

### Create 'core groups'

After getting the route durations using an API, the network built in the first step is modified so that there are at least that many connedted subgraph as 'stay' dictates. Also, only 75% (MAX_CORE_GROUP_COVERAGE) of all locations should be part of a subgraph (better results this way).

<img width="500" height="767" alt="Step2" src="https://github.com/user-attachments/assets/61315ae2-2b30-441c-823a-d59233668fa6" />

### Modify 'core groups' & calculate routes

If the core group is too small (sum of route durations + time spent at locations < duration of day (MAX_TOTAL_WAYLENGTH)): Extend subgraph with graphs / locations 'on the way' or 'near' the group

<img width="500" height="767" alt="Step3a" src="https://github.com/user-attachments/assets/2ae35de1-217c-4894-ba17-e3e27149b5ed" />

The optimal route within the subgraph is found using TSP. The result is also the condition that decides which of the opions above and below is true and therefore is repeated mutiple times.

<img width="500" height="767" alt="Setp3b" src="https://github.com/user-attachments/assets/0e3b4847-36bd-4384-9549-a6d1b584f878" />

If the day trip is too large, remove routes that 'bulge' the route until the duration of one day is reached.

<img width="500" height="749" alt="Setp3c" src="https://github.com/user-attachments/assets/6ff21995-df44-4cf3-8c53-bff97dcaeffc" />
