#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <algorithm>

using namespace std;

typedef unsigned int uint;
typedef unsigned long long ull;

#define BLOCK_SIZE 64
const double ratio = 0.9;

__global__ void TC_kernal(ull *d_cnt, int *d_col, int *d_off, int *d_row, int *d_map, int map_stride, int row_idx)
{
    ull cnt = 0;
    for (int i = blockIdx.x; i < row_idx; i += gridDim.x) {
        int st = d_off[d_row[i]];
        int ed = d_off[d_row[i]+1];
        for (int j = st; j < ed; j += blockDim.x) {
            int u;
            if (j + threadIdx.x < ed) {
                u = d_col[j+threadIdx.x];
                atomicOr(d_map + blockIdx.x*map_stride + (u >> 5), (1 << (u & 0x1f)));
            }
            __syncthreads();
            int ked = min(ed-j, blockDim.x);
            for (int k = 0; k < ked; k++) {
                int t = d_col[j+k];
                int l = d_off[t] + threadIdx.x;
                int r = d_off[t+1];
                while (l < r) {
                    int v = d_col[l];
                    if (d_map[blockIdx.x*map_stride + (v >> 5)] & (1 << (v & 0x1f))) cnt++;
                    l += blockDim.x;
                }
            }
            __syncthreads();
        }
        for (int j = st; j < ed; j += blockDim.x) {
            if (j + threadIdx.x < ed) {
                int u = d_col[j+threadIdx.x];
                d_map[blockIdx.x*map_stride + (u >> 5)] = 0;
            }
        }
        __syncthreads();
    }

    __shared__ ull tot[BLOCK_SIZE];
    tot[threadIdx.x] = cnt;
    __syncthreads();
    for (int k = blockDim.x >> 1; k; k >>= 1) {
        if (threadIdx.x < k) {
            tot[threadIdx.x] += tot[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) 
        d_cnt[blockIdx.x] = tot[0];
}

union edge
{
    uint a[2];
    ull l;
    edge(){}
    edge(uint _u, uint _v){
        a[0] = _u;
        a[1] = _v;
    }
    bool operator <(const edge r) const {
        return l < r.l;
    }
    bool operator ==(const edge r) const {
        return l == r.l;
    }
};

int *col, *off, *row, *hash, *bitset;
int *d_col, *d_off, *d_row, *d_map;
ull *h_cnt, *d_cnt, *h_sort, *d_sort;
int col_idx, row_idx, off_idx, edge_idx, point_idx;
edge *edges;
ull ans = 0;

int cri(int u)
{
    #pragma omp critical(a) 
    {
        if (hash[u] == 0) u = hash[u] = point_idx++;
        else u = hash[u];
    }
    return u;
}

int main(int argc, char const *argv[])
{
    cudaDeviceReset();

    char filename[100];
    if (argc > 1) {
        int i = 1;
        while (argv[i][0] == '-') i++;
        strcpy(filename, argv[i]);
    } else {
        // strcpy(filename, "/data/soc-LiveJournal1.bin");
        // strcpy(filename, "/data/s24.kron.edgelist");
        // strcpy(filename, "/data/twitter_rv.bin");
        // strcpy(filename, "/data/s26.kron.edgelist");
        strcpy(filename, "/data/s27.kron.edgelist");
    }

    struct stat statbuf;
    stat(filename,&statbuf);
    size_t size = statbuf.st_size;

    cudaHostAlloc((void **)&edges, size, cudaHostAllocMapped);
    edge_idx = 0;

    FILE *fin = fopen(filename, "rb");
    int wid = 256;
    int fid = 256;
    while (fid == wid) {
        fid = fread(edges + edge_idx, 4, wid, fin);
        int ed = edge_idx + (fid >> 1);
        for (int i = edge_idx; i < ed; i++) 
            if (edges[i].a[0] != edges[i].a[1]) {
                if (i == edge_idx) {
                    edge_idx++;
                } else {
                    edges[edge_idx++].l = edges[i].l;
                }
            }
    }
    fclose(fin);

    hash = (int *)malloc(sizeof(int)*(1lu<<32));
    memset(hash, 0, sizeof(int)*(1lu<<32));
    point_idx = 1;

    #pragma omp parallel for
    for (int i = 0; i < edge_idx; i++) {
        int u = edges[i].a[0];
        int v = edges[i].a[1];
        int t = hash[u];
        if (t != 0) {
            u = t;
        } else if (u != 0) {
            u = cri(u);
        } 
        t = hash[v];
        if (t != 0) {
            v = t;
        } else if (v != 0) {
            v = cri(v);
        }
        if (u > v) {
            t = u;
            u = v;
            v = t;
        }
        edges[i].a[0] = u;
        edges[i].a[1] = v;
    }

    cudaHostGetDevicePointer((void **)&d_sort, (void *)edges, 0);
    thrust::sort((ull *)d_sort, (ull *)(d_sort + edge_idx));
    edge_idx = unique(edges, edges + edge_idx) - edges;

    col = (int *)malloc(sizeof(int)*edge_idx);
    off = (int *)malloc(sizeof(int)*(point_idx+2));
    row = (int *)malloc(sizeof(int)*point_idx);

    int last = -1;
    off_idx = row_idx = 0;
    for (int i = 0; i < edge_idx; i++) {
        int u = edges[i].a[0];
        int v = edges[i].a[1];
        col[i] = u;
        while (off_idx <= v)
            off[off_idx++] = i;
        if (v != last)
            row[row_idx++] = v;
        last = v;
    }
    off[off_idx++] = edge_idx;
    off[off_idx++] = edge_idx;

    int row_gpu_en;
    ull prefix_sum = 0;
    for (int i = 0; i < row_idx; i++) {
        ull m = off[row[i]+1] - off[row[i]];
        prefix_sum += m;
        if (prefix_sum > edge_idx * ratio) {
            row_gpu_en = i;
            break;
        }
    }

    cudaMalloc((void **) &d_col, sizeof(int)*edge_idx);
    cudaMalloc((void **) &d_off, sizeof(int)*off_idx);
    cudaMalloc((void **) &d_row, sizeof(int)*row_idx);

    // Rest 1G for GPU
    size_t mem_tot;
    size_t mem_free;
    cudaMemGetInfo(&mem_free, &mem_tot);
    int map_stride = (off_idx >> 5) + 1;
    int grid_size = (mem_free - (1<<30)) / (sizeof(int)*map_stride + sizeof(ull));

    h_cnt = (ull *)malloc(sizeof(ull)*grid_size);
    cudaMalloc((void **) &d_map, sizeof(int)*map_stride*grid_size);
    cudaMalloc((void **) &d_cnt, sizeof(ull)*grid_size);
    cudaMemcpy(d_col, col, sizeof(int)*edge_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, off, sizeof(int)*off_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, row, sizeof(int)*row_idx, cudaMemcpyHostToDevice);
    cudaMemset(d_map, 0, sizeof(int)*map_stride*grid_size);


    TC_kernal<<< grid_size, BLOCK_SIZE >>>(d_cnt, d_col, d_off, d_row, d_map, map_stride, row_gpu_en);

    bitset = (int *)malloc(sizeof(int)*map_stride*omp_get_num_procs());
    memset(bitset, 0, sizeof(int)*map_stride*omp_get_num_procs());
    #pragma omp parallel for reduction(+:ans) shared(bitset) schedule(dynamic)
    for (int i = row_gpu_en; i < row_idx; i++) {
        int thread_id = omp_get_thread_num();
        int st = off[row[i]];
        int ed = off[row[i]+1];
        for (int j = st; j < ed; j++) {
            int to = col[j];
            bitset[thread_id*map_stride + (to >> 5)] |= (1 << (to & 0x1f));
        }
        for (int j = st; j < ed; j++) {
            int t1 = col[j];
            int kst = off[t1];
            int ked = off[t1+1];
            for (int k = kst; k < ked; k++) {
                int t2 = col[k];
                if ( bitset[thread_id*map_stride + (t2 >> 5)] & (1 << (t2 & 0x1f)) ) ans++;
            }
        }
        for (int j = st; j < ed; j++) {
            bitset[thread_id*map_stride + (col[j] >> 5)] = 0;
        }
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_cnt, d_cnt, sizeof(ull)*grid_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < grid_size; i++)
        ans += h_cnt[i];
    printf("There are %llu triangles in the input graph.\n", ans);

    free(col);
    free(off);
    free(row);
    free(hash);
    free(h_cnt);
    free(bitset);
    cudaFree(d_col);
    cudaFree(d_off);
    cudaFree(d_row);
    cudaFree(d_cnt);
    cudaFree(d_map);
    cudaFreeHost(edges);
    return 0;
}
