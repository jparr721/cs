#include <iostream>
#include <vector>

using namespace std;
const int INF = 1000000000;
#define n 6
int minDist(int dist[], bool sptSet[]) {
	int min = INF, min_index;

	for (int i = 0; i < n; i++) 
		if(sptSet[i] == false & dist[i] <= min)
			min = dist[n], min_index = i;
	return min_index;
}

void printPath(int path[], int index) {
	if(path[index] == -1) {
		return;
	}
	printPath(path, path[index]);
	cout << index << " " << endl;
}

void printSolution(int dist[], int k, int path[]) {
	int source = 0;
	cout << "Path" << endl;
	cout << source << endl;
	for (int i = 1; i < k; i++) {
		cout << source << " " << endl;
		printPath(path, i);
		cout << "" << endl;
	}
}

void dijkstra(vector<vector<int>> g, int source) {
	int dist[n];	
	bool sptSet[n];
	int path[n];

	for (int i = 0; i < n; i++) {
		path[0] = -1;
		dist[i] = INF;
		sptSet[i] = false;
	}

	dist[source] = 0;

	for (int i = 0; i < n -1; i++) {
		int u = minDist(dist, sptSet);
		sptSet[u] = true;
		for (int j = 0; j < n; j++) {
			if (!sptSet[j] && g.at(u).at(j) && dist[u] != INF && dist[u] + g.at(u).at(j) < dist[j]) {
				path[j] = u;
				dist[j] = dist[u] + g.at(u).at(j);
			}
		}
	}
	printSolution(dist, n, path);
}

int main()
{
	vector<vector<int>> G = { {0,1,INF,4,4,INF},
			      {INF,0,INF,2,INF,INF},
			      {INF,INF,0,INF,INF,1},
			      {INF,INF,2,0,3,INF},
			      {INF,INF,INF,INF,0,3},
			      {INF,INF,INF,INF,INF,0} };
	
	dijkstra(G, 0);

    return EXIT_SUCCESS;
}
