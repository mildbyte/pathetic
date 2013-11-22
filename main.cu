#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_functions.h>
#include <math_constants.h>
#include <stdio.h>
#include <limits>
#include <conio.h>

struct Sphere {
	float3 position;
	float radius;
	float3 colour;
	float3 emission;
};

struct Ray {
	float3 origin;
	float3 direction;
};

struct Scene {
	int size;
	Sphere* spheres;
};

struct Bitmap {
	int width;
	int height;
	int stride;
	float3 *elements;
};

float3 getElement(const Bitmap b, int row, int col) {
	return b.elements[row * b.stride + col];
}

__device__ void setElement(Bitmap b, int row, int col, float3 value) {
	b.elements[row * b.stride + col] = value;
}

__device__ bool intersect(Ray r, Sphere s, float &distance) {
	float dx = r.origin.x - s.position.x;
	float dy = r.origin.y - s.position.y;
	float dz = r.origin.z - s.position.z;

	float b = 2.0f * (
		  r.direction.x * dx
		+ r.direction.y * dy
		+ r.direction.z * dz);
	float c = dx * dx + dy * dy + dz * dz - s.radius * s.radius;
	float d = b * b - 4.0f * c;

	if (d < 0.0f) return false;

	float sqrtd = sqrtf(d);

	distance = (-b - sqrtd) * 0.5f;
	if (distance > 0.0f) return true;

	distance = (-b + sqrtd) * 0.5f;
	if (distance > 0.0f) return true;
	return false;
}

__device__ int sceneIntersect(Ray r, Scene scene, float &closest_dist) {
	int closest_id = -1;
	closest_dist = FLT_MAX;

	for (int i = 0; i < scene.size; i++) {
		float dist;
		if (!intersect(r, scene.spheres[i], dist)) continue;

		if (dist < closest_dist) {
			closest_dist = dist;
			closest_id = i;
		}
	}

	return closest_id;
}

__device__ float3 traceRay(Ray r, Scene scene) {
	float3 black = {0.0f, 0.0f, 0.0f};

	float dist;
	int id = sceneIntersect(r, scene, dist);

	if (id == -1) return black;
	return scene.spheres[id].colour;
}

__device__ inline float3 normalize(float3 v) {
	float invdist = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	float3 result = {
		v.x * invdist,
		v.y * invdist,
		v.z * invdist
	};

	return result;
}

__global__ void rayTrace(Scene scene, Bitmap bitmap, float3 cameraPos, float3 imagePlaneCentre, float3 xPixel, float3 yPixel) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= bitmap.width || y >= bitmap.height) return;

	//float3 black = {0.0f, 0.0f, 0.0f};
	//setElement(bitmap, y, x, black);

	float xFactor = x - 0.5f * bitmap.width + 0.5f;
	float yFactor = y - 0.5f * bitmap.height + 0.5f;
	
	float3 rayDir = {
		imagePlaneCentre.x + xPixel.x * xFactor + yPixel.x * yFactor,
		imagePlaneCentre.y + xPixel.y * xFactor + yPixel.y * yFactor,
		imagePlaneCentre.z + xPixel.z * xFactor + yPixel.z * yFactor,
	};

	rayDir = normalize(rayDir);

	//float3 rayDir = {0.0f, 0.0f, 1.0f};

	Ray ray;
	ray.origin = cameraPos;
	ray.direction = rayDir;

	setElement(bitmap, y, x, traceRay(ray, scene));
}

int clamp(float component) {
	int c = (int)roundf(component * 255.0f);
	if (c > 255) c = 255;
	if (c < 0) c = 0;

	return c;
}

void saveBitmapToFile(Bitmap bitmap, char *filename) {
	//Bitmap saving code inspired by
    //http://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries

	int filesize = 3 * bitmap.width * bitmap.height;
    unsigned char bfheader [14] = {'B','M',0,0,0,0,0,0,0,0,54,0,0,0};
    unsigned char biheader [40] = {40,0,0,0,0,0,0,0,0,0,0,0,1,0,24,0};
    
    bfheader[ 2] = (unsigned char)(filesize    );
	bfheader[ 3] = (unsigned char)(filesize>> 8);
	bfheader[ 4] = (unsigned char)(filesize>>16);
	bfheader[ 5] = (unsigned char)(filesize>>24);

	biheader[ 4] = (unsigned char)(bitmap.width);
	biheader[ 5] = (unsigned char)(bitmap.width>> 8);
	biheader[ 6] = (unsigned char)(bitmap.width>>16);
	biheader[ 7] = (unsigned char)(bitmap.width>>24);
	biheader[ 8] = (unsigned char)(bitmap.height    );
	biheader[ 9] = (unsigned char)(bitmap.height>> 8);
	biheader[10] = (unsigned char)(bitmap.height>>16);
	biheader[11] = (unsigned char)(bitmap.height>>24);
    
    //Open the output file and write the header
    FILE* output;

    output = fopen(filename, "wb");

    fwrite(&bfheader, 1, sizeof(bfheader), output);
    fwrite(&biheader, 1, sizeof(biheader), output);

    //Output the bitmap

    //BMP requires every row to be padded to 4 bytes
    int padding = 4 - ((bitmap.width * 3) % 4);
    if (padding == 4) padding = 0; //side effect if width mod 4 = 0 :)

	for (int i = bitmap.height - 1; i >= 0; i--) {
		for (int j = 0; j < bitmap.width; j++) {
			//Write the B, G, R to the output
            unsigned char clamped;
			clamped = clamp(getElement(bitmap, i, j).z);
            fwrite(&clamped, 1, 1, output);

            clamped = clamp(getElement(bitmap, i, j).y);
            fwrite(&clamped, 1, 1, output);

            clamped = clamp(getElement(bitmap, i, j).x);
            fwrite(&clamped, 1, 1, output);
        }
        
        //Pad the row to 4 bytes
        for (int p = 0; p < padding; p++) fputc(0, output);
    }

	fclose(output);
}

int main() {
	Scene testScene;
	testScene.size = 2;
	testScene.spheres = new Sphere[2];
	testScene.spheres[0].position.x = 0.0f;
	testScene.spheres[0].position.y = 0.0f;
	testScene.spheres[0].position.z = 10.0f;
	testScene.spheres[0].radius = 5.0f;
	testScene.spheres[0].colour.x = 0.0f;
	testScene.spheres[0].colour.y = 1.0f;
	testScene.spheres[0].colour.z = 0.0f;
	testScene.spheres[1].position.x = 3.0f;
	testScene.spheres[1].position.y = 1.0f;
	testScene.spheres[1].position.z = 10.0f;
	testScene.spheres[1].radius = 4.0f;
	testScene.spheres[1].colour.x = 1.0f;
	testScene.spheres[1].colour.y = 0.0f;
	testScene.spheres[1].colour.z = 0.0f;

	Scene deviceScene = testScene;

	int resX = 640;
	int resY = 480;

	Bitmap deviceBitmap;
	deviceBitmap.height = resY;
	deviceBitmap.width = resX;
	deviceBitmap.stride = resX;

	float3 cameraPos = {0.0f, 0.0f, -10.0f};
	float3 cameraToImagePlane = {0.0f, 0.0f, 10.0f};
	float3 xPixel = {16.0f/resX, 0.0f, 0.0f};
	float3 yPixel = {0.0f, -12.0f/resY, 0.0f};

	dim3 threadsPerBlock(16, 16);

	dim3 numBlocks(resX / threadsPerBlock.x + (resX % threadsPerBlock.x == 0? 0 : 1),
		resY / threadsPerBlock.y + (resY % threadsPerBlock.y == 0? 0 : 1));

	cudaError_t err;

	printf("Allocating the array on the CUDA device...\n");
	cudaMalloc(&(deviceScene.spheres), testScene.size * sizeof(Sphere));
	cudaMalloc(&(deviceBitmap.elements), resX * resY * sizeof(float3));
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while allocating memory on the GPU!\n");
		return -1;
	}

	printf("Copying the scene to the CUDA device...\n");
	cudaMemcpy(deviceScene.spheres, testScene.spheres, testScene.size * sizeof(Sphere), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while copying the scene!\n");
		return -1;
	}

	printf("Starting the kernels...\n");
	rayTrace<<<numBlocks, threadsPerBlock>>>(deviceScene, deviceBitmap, cameraPos, cameraToImagePlane, xPixel, yPixel);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while executing the child kernel!\n");
		return -1;
	}

	//...and now we wait.
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while executing the child kernel!\n");
		return -1;
	}

	printf("Job finished, copying the buffer back...\n");

	Bitmap localBitmap = deviceBitmap;
	localBitmap.elements = new float3[resX * resY];

	cudaMemcpy(localBitmap.elements, deviceBitmap.elements, sizeof(float3) * resX * resY, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) {
		printf("wtf");
		return -1;
	}

	printf("we found something!\n");

	saveBitmapToFile(localBitmap, "test.bmp");

	printf("Saving complete.\n");

	getch();
}