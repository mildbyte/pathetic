#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <math_functions.h>
#include <math_constants.h>
#include <stdio.h>
#include <limits>
#include <conio.h>

/* TODO:
 * Add comments
 * use __shared__ for blocks
 * split into modules (figure out the structure)
 * port camera->world code
 * clipping the image plane
 * write a readme
 */

struct Sphere {
	float3 position;
	float radius;
	float3 colour;
	float3 emission;

	Sphere() {
		position = make_float3(0.0f, 0.0f, 0.0f);
		colour = make_float3(0.0f, 0.0f, 0.0f);
		emission = make_float3(0.0f, 0.0f, 0.0f);
		radius = 0.0f;
	}
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


__device__ inline float3 normalize(float3 v) {
	float invdist = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	float3 result = {
		v.x * invdist,
		v.y * invdist,
		v.z * invdist
	};

	return result;
}

__device__ float3 hemisphereSample(float3 normal, float &dot, curandState_t *state) {
	float3 result;
	do {
		result.x = curand_uniform(state) * 2.0f - 1.0f;
		result.y = curand_uniform(state) * 2.0f - 1.0f;
		result.z = curand_uniform(state) * 2.0f - 1.0f;

		dot = result.x * normal.x
		+ result.y * normal.y
		+ result.z * normal.z;
	} while (dot > 1.0f || dot <= 0.0f);

	return normalize(result);
}

__device__ float3 traceRay(Ray ray, Scene scene, int level, curandState_t *state) {
	float3 result = {0.0f, 0.0f, 0.0f};
	float3 factor = {1.0f, 1.0f, 1.0f};

	for (int i = 0; i <= level; i++) {
		float dist;
		int id = sceneIntersect(ray, scene, dist);

		if (id == -1) break;
		Sphere sphere = scene.spheres[id];

		ray.origin = make_float3(
			ray.origin.x + ray.direction.x * dist,
			ray.origin.y + ray.direction.y * dist,
			ray.origin.z + ray.direction.z * dist);

		float3 normal = {
				ray.origin.x - sphere.position.x,
				ray.origin.y - sphere.position.y,
				ray.origin.z - sphere.position.z
		};

		normal = normalize(normal);

		float dot;
		ray.direction = hemisphereSample(normal, dot, state);
		
		ray.origin.x += ray.direction.x * 0.001;
		ray.origin.y += ray.direction.y * 0.001;
		ray.origin.z += ray.direction.z * 0.001;

		result.x += factor.x * sphere.emission.x;
		result.y += factor.y * sphere.emission.y;
		result.z += factor.z * sphere.emission.z;

		factor.x *= sphere.colour.x * dot;
		factor.y *= sphere.colour.y * dot;
		factor.z *= sphere.colour.z * dot;
	}
	
	return result;
}

__global__ void setupPRNGs(curandState *state) {
	int id = threadIdx.x + blockDim.x * threadIdx.y;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void rayTrace(Scene scene, Bitmap bitmap, float3 cameraPos, float3 imagePlaneCentre, float3 xPixel, float3 yPixel, int noSamples, curandState *states) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = threadIdx.x + blockDim.x * threadIdx.y;

	curandState state = states[id];

	if (x >= bitmap.width || y >= bitmap.height) return;

	float xFactor = x - 0.5f * bitmap.width + 0.5f;
	float yFactor = y - 0.5f * bitmap.height + 0.5f;
	
	float3 rayDir = {
		imagePlaneCentre.x + xPixel.x * xFactor + yPixel.x * yFactor,
		imagePlaneCentre.y + xPixel.y * xFactor + yPixel.y * yFactor,
		imagePlaneCentre.z + xPixel.z * xFactor + yPixel.z * yFactor,
	};

	rayDir = normalize(rayDir);

	Ray ray;
	ray.origin = cameraPos;
	ray.direction = rayDir;

	float3 result = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < noSamples; i++) {
		float3 sample = traceRay(ray, scene, 5, &state);
		result.x += sample.x;
		result.y += sample.y;
		result.z += sample.z;
	}

	float factor = 1.0f/noSamples;

	result.x *= factor;
	result.y *= factor;
	result.z *= factor;

	setElement(bitmap, y, x, result);
}

int clamp(float component) {
	component = component / (1.0f + component);
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
	testScene.spheres[0].position.x = -5.0f;
	testScene.spheres[0].position.y = 0.0f;
	testScene.spheres[0].position.z = 10.0f;
	testScene.spheres[0].radius = 5.0f;
	testScene.spheres[0].colour.x = 1.0f;
	testScene.spheres[0].colour.y = 0.0f;
	testScene.spheres[0].colour.z = 0.0f;
	testScene.spheres[0].emission.x = 1.0f;
	testScene.spheres[0].emission.y = 0.5f;
	testScene.spheres[0].emission.z = 0.5f;
	testScene.spheres[1].position.x = 5.0f;
	testScene.spheres[1].position.y = 0.0f;
	testScene.spheres[1].position.z = 10.0f;
	testScene.spheres[1].radius = 5.0f;
	testScene.spheres[1].colour.x = 1.0f;
	testScene.spheres[1].colour.y = 1.0f;
	testScene.spheres[1].colour.z = 1.0f;

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

	curandState *prngStates;

	cudaError_t err;

	printf("Allocating the array on the CUDA device...\n");
	cudaMalloc(&(deviceScene.spheres), testScene.size * sizeof(Sphere));
	cudaMalloc(&(deviceBitmap.elements), resX * resY * sizeof(float3));
	cudaMalloc(&prngStates, sizeof(curandState) * threadsPerBlock.x * threadsPerBlock.y);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while allocating memory on the GPU!\n");
		return -1;
	}

	printf("Initialising the random number generators...\n");
	setupPRNGs<<<numBlocks, threadsPerBlock>>>(prngStates);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while seeding the PRNGs!\n");
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
	rayTrace<<<numBlocks, threadsPerBlock>>>(deviceScene, deviceBitmap, cameraPos, cameraToImagePlane, xPixel, yPixel, 100, prngStates);
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
		printf("Error while retrieving the bitmap!");
		return -1;
	}

	printf("Tonemapping and saving...\n");

	saveBitmapToFile(localBitmap, "test.bmp");

	printf("Saving complete.\n");
	cudaDeviceReset();
}