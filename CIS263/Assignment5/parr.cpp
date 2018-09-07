#include <iostream>
#include <vector>
#include <limits.h>

using namespace std;
const int INF = 1000000000;

void dijkstra(vector<vector<int> > graph, int src);
int minDistance(int dist[], bool sptSet[], int size);
void printPath(int parent[], int j);
void printSolution(int dist[], int size, int parent[]);


int main()
{
    int n = 6;  //number of vertices

    vector<vector<int> > G = { {0,1,INF,4,4,INF},
                              {INF,0,INF,2,INF,INF},
                              {INF,INF,0,INF,INF,1},
                              {INF,INF,2,0,3,INF},
                              {INF,INF,INF,INF,0,3},
                              {INF,INF,INF,INF,INF,0} };

    dijkstra(G, 0);
    return 1;
}

void dijkstra(vector<vector<int> > graph, int src){
    int size = graph.size();
    int dist[size];
    int parent[size];
    bool sptSet[size];
    
    for(int i=0; i<size; i++){
        parent[0] = -1;
        dist[i] = INF;
        sptSet[i] = false;
    }

    dist[src] = 0;

    for(int count = 0; count < size - 1; count++){
        int u = minDistance(dist, sptSet, size);
        sptSet[u] = true;
        for(int v = 0; v < size; v++){
            if(!sptSet[v] && graph.at(u).at(v) && dist[u] != INT_MAX && dist[u] + graph.at(u).at(v) < dist[v]){
                parent[v] = u;
                dist[v] = dist[u] + graph.at(u).at(v);
            }
        }
    }

    printSolution(dist, size, parent);
}

void printPath(int parent[], int j){
    if(parent[j] == -1){
        return;
    }
    printPath(parent, parent[j]);
    cout << j << " "; 

}

void printSolution(int dist[], int n, int parent[]){
    int src = 0;
    cout << "Path" << endl;
    cout << src << endl;
    for (int i = 1; i < n; i++)
    {
        cout << src << " " ;
        printPath(parent, i);
        cout << endl;
    }
}

int minDistance(int dist[], bool sptSet[], int size){
    int min = INT_MAX, min_index;

    for(int i = 0; i < size; i++){
        if(sptSet[i] == false && dist[i] <= min){
            min = dist[i], min_index = i;
        }
    }
    return min_index;
}


