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

	Sphere(float3 p, float r, float3 c, float3 e) {
		position = p;
		radius = r;
		colour = c;
		emission = e;
	}
};

//A scene so far is just an array of spheres.
struct Scene {
	int size;
	Sphere* spheres;
};

//Represents an unnormalized bitmap.
struct Bitmap {
	int width;
	int height;
	int stride;
	float3 *elements;
};

__device__ inline float3 getElement(const Bitmap b, int row, int col) {
	return b.elements[row * b.stride + col];
}

float3 hostGetElement(const Bitmap b, int row, int col) {
	return b.elements[row * b.stride + col];
}

__device__ inline void setElement(Bitmap b, int row, int col, float3 value) {
	b.elements[row * b.stride + col] = value;
}

//Returns true and sets the distance if the ray this thread is responsible for 
//intersects this sphere, false otherwise (and the distance is undefined then).
__device__ bool intersect(int threadId, Sphere s, float &distance) {
	//The ray's position and direction are stored in rayBuffer[threadId * 2] 
	//and rayBuffer[threadId * 2 + 1], respectively.
	extern __shared__ float3 rayBuffer[];

	//Find the discriminant of the quadratic equation (standard intersection test)
	float dx = rayBuffer[threadId * 2].x - s.position.x;
	float dy = rayBuffer[threadId * 2].y - s.position.y;
	float dz = rayBuffer[threadId * 2].z - s.position.z;

	float b = 2.0f * (
		  rayBuffer[threadId * 2 + 1].x * dx
		+ rayBuffer[threadId * 2 + 1].y * dy
		+ rayBuffer[threadId * 2 + 1].z * dz);

	float d = b * b - 4.0f * (dx * dx + dy * dy + dz * dz - s.radius * s.radius);

	if (d < 0.0f) return false; //The ray misses the sphere

	//Otherwise, try the closer distance...
	float sqrtd = sqrtf(d);

	//If it's behind the camera (not the image plane for now), try the further distance...
	distance = (-b - sqrtd) * 0.5f;
	if (distance > 0.0f) return true;

	//If even this distance is behind the camera, the ray misses the sphere.
	distance = (-b + sqrtd) * 0.5f;
	if (distance > 0.0f) return true;
	return false;
}

//Goes through the scene and finds the closest sphere that intersects the ray
//this thread is responsible for. If it's found, returns its id and sets the
//closest distance variable, if not, returns -1.
__device__ int sceneIntersect(int threadId, Scene scene, float &closest_dist) {
	int closest_id = -1;
	closest_dist = FLT_MAX;

	for (int i = 0; i < scene.size; i++) {
		float dist;
		if (!intersect(threadId, scene.spheres[i], dist)) continue;

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

//Returns a vector in the hemisphere around the normal and sets dot to the dot product
//of the result with the normal (using rejection sampling).
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

//Does the pathtracing of a ray recorded in the shared memory, terminates after <level> bounces.
__device__ float3 traceRay(int threadId, Scene scene, int level, curandState_t *state) {
	extern __shared__ float3 rayBuffer[];

	float3 result = {0.0f, 0.0f, 0.0f};	//Current accumulated pixel colour
	float3 factor = {1.0f, 1.0f, 1.0f}; //Accumulated colours of the oblects
	//this ray has hit (product), every contribution is multiplied by this
	//factor before being added to result.

#pragma unroll 128
	for (int i = 0; i <= level; i++) {
		//If nothing has been hit, terminate the procedure.
		//TODO: don't trace the initial ray (it's done at every sampling otherwise),
		//pass to the routine the actual point of intersection.
		float dist;
		int id = sceneIntersect(threadId, scene, dist);

		if (id == -1) break;
		Sphere sphere = scene.spheres[id];

		//Set the ray's position to the position of the collision with the sphere
		//(position + direction * distance).
		rayBuffer[threadId * 2] = make_float3(
			rayBuffer[threadId * 2].x + rayBuffer[threadId * 2 + 1].x * dist,
			rayBuffer[threadId * 2].y + rayBuffer[threadId * 2 + 1].y * dist,
			rayBuffer[threadId * 2].z + rayBuffer[threadId * 2 + 1].z * dist);

		//Find the normal to the collision point and normalize it.
		float3 normal = {
				rayBuffer[threadId * 2].x - sphere.position.x,
				rayBuffer[threadId * 2].y - sphere.position.y,
				rayBuffer[threadId * 2].z - sphere.position.z
		};

		normal = normalize(normal);

		//Take a random direction in the hemisphere around this normal.
		float dot;
		rayBuffer[threadId * 2 + 1] = hemisphereSample(normal, dot, state);
		
		//Perform the "epsilon shift" so that the ray doesn't collide with itself.
		rayBuffer[threadId * 2].x += rayBuffer[threadId * 2 + 1].x * 0.001;
		rayBuffer[threadId * 2].y += rayBuffer[threadId * 2 + 1].y * 0.001;
		rayBuffer[threadId * 2].z += rayBuffer[threadId * 2 + 1].z * 0.001;

		//Add the emitted light strength to the result.
		result.x += factor.x * sphere.emission.x;
		result.y += factor.y * sphere.emission.y;
		result.z += factor.z * sphere.emission.z;

		//The next contributions will be multiplied by the cosine of the diffusion angle
		//and the colour of the sphere.
		factor.x *= sphere.colour.x * dot;
		factor.y *= sphere.colour.y * dot;
		factor.z *= sphere.colour.z * dot;
	}
	
	return result;
}

//Leaves pseudorandom generator states in the device memory.
//TODO: bake randoms?
__global__ void setupPRNGs(curandState *state) {
	int id = threadIdx.x + blockDim.x * threadIdx.y;
	curand_init(1234, id, 0, &state[id]);
}

//The main raytracing kernel. Generates the initial ray and passes it to the pathtracer.
__global__ void rayTrace(Scene scene, Bitmap bitmap, float3 cameraPos, float3 imagePlaneCentre, float3 xPixel, float3 yPixel, curandState *states) {
	extern __shared__ float3 rayBuffer[];

	//Which pixel are we responsible for?
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Which thread are we?
	int id = threadIdx.x + blockDim.x * threadIdx.y;

	//If we are out of the bounds of the image, stop (when eg. the block size
	//doesn't divide the image size).
	if (x >= bitmap.width || y >= bitmap.height) return;

	//Shift the coordinates because we keep track of the centre of
	//the image plane, not its corner (and we're sampling the centres
	//of the pixels).
	//TODO: maybe jitter the position of the sampling point every sample?
	float xFactor = x - 0.5f * bitmap.width + 0.5f;
	float yFactor = y - 0.5f * bitmap.height + 0.5f;
	
	//The rays are stored in a shared memory space, visible to all threads in the block.
	//The origin is stored at id * 2 and the direction - at id * 2 + 1. This is done because
	//the shared memory is faster than the thread local memory, and the thread needs to access
	//the ray quite often.

	//Construct the vector from the camera to the centre of the pixel on the image plane
	//and normalize it.
	rayBuffer[id * 2 + 1] = normalize(make_float3(
		imagePlaneCentre.x + xPixel.x * xFactor + yPixel.x * yFactor,
		imagePlaneCentre.y + xPixel.y * xFactor + yPixel.y * yFactor,
		imagePlaneCentre.z + xPixel.z * xFactor + yPixel.z * yFactor));

	//TODO: can we optimize this out?
	rayBuffer[id * 2] = cameraPos;

	//Take one illumination sample from the scene and add it to the bitmap.
	float3 current = getElement(bitmap, y, x);
	float3 sample = traceRay(id, scene, 5, &states[id]);
	current.x += sample.x;
	current.y += sample.y;
	current.z += sample.z;

	setElement(bitmap, y, x, current);
}

//Multiplies all values in the bitmap by a certain factor (used for normalizing
//after the sampling, multiplied by 1/noSamples).
__global__ void scale(Bitmap bitmap, float factor) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= bitmap.width || y >= bitmap.height) return;

	float3 pixel = getElement(bitmap, y, x);
	pixel.x *= factor;
	pixel.y *= factor;
	pixel.z *= factor;

	setElement(bitmap, y, x, pixel);
}

//Initializes the bitmap with zeroes.
__global__ void zeroBitmap(Bitmap bitmap) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= bitmap.width || y >= bitmap.height) return;

	float3 pixel = {0.0f, 0.0f, 0.0f};

	setElement(bitmap, y, x, pixel);
}

//Tonemaps a component and converts it to 0..255.
int clamp(float component) {
	component = component / (1.0f + component);
	int c = (int)roundf(component * 255.0f);
	if (c > 255) c = 255;
	if (c < 0) c = 0;

	return c;
}

//Saves a bitmap to a BMP file.
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
			clamped = clamp(hostGetElement(bitmap, i, j).z);
			fwrite(&clamped, 1, 1, output);

			clamped = clamp(hostGetElement(bitmap, i, j).y);
			fwrite(&clamped, 1, 1, output);

			clamped = clamp(hostGetElement(bitmap, i, j).x);
			fwrite(&clamped, 1, 1, output);
		}
        
		//Pad the row to 4 bytes
		for (int p = 0; p < padding; p++) fputc(0, output);
	}

	fclose(output);
}

int main() {
	Scene testScene;
	testScene.size = 4;
	testScene.spheres = new Sphere[testScene.size];
	
	//Simulate planes with very large spheres.
	testScene.spheres[0] = Sphere(make_float3(0, 1000, 0), 1000, make_float3(0.5, 1.0, 0.5), make_float3(0, 0, 0));
	testScene.spheres[1] = Sphere(make_float3(0, 0, 1020), 1000, make_float3(1, 1, 1), make_float3(0, 0, 0));
	testScene.spheres[2] = Sphere(make_float3(0, -5, 10), 5, make_float3(0.5, 0.5, 0.9), make_float3(0.0, 0.0, 0.0));
	testScene.spheres[3] = Sphere(make_float3(-5, -20, 20), 5, make_float3(0.5, 0.5, 0.5), make_float3(10.0, 10.0, 7.0));

	Scene deviceScene = testScene;

	int resX = 640;
	int resY = 480;

	Bitmap deviceBitmap;
	deviceBitmap.height = resY;
	deviceBitmap.width = resX;
	deviceBitmap.stride = resX;

	float3 cameraPos = {0.0f, -5.0f, -10.0f};
	float3 cameraToImagePlane = {0.0f, 0.0f, 10.0f};
	float3 xPixel = {16.0f/resX, 0.0f, 0.0f};
	float3 yPixel = {0.0f, 12.0f/resY, 0.0f};

	dim3 threadsPerBlock(16, 16);

	dim3 numBlocks(resX / threadsPerBlock.x + (resX % threadsPerBlock.x == 0? 0 : 1),
		resY / threadsPerBlock.y + (resY % threadsPerBlock.y == 0? 0 : 1));

	curandState *prngStates;

	cudaError_t err;

	//Allocate the memory for the scene, the bitmap, the random generator state
	//and initialize the bitmap.
	printf("Allocating the array on the CUDA device...\n");
	cudaMalloc(&(deviceScene.spheres), testScene.size * sizeof(Sphere));
	cudaMalloc(&(deviceBitmap.elements), resX * resY * sizeof(float3));
	zeroBitmap<<<numBlocks, threadsPerBlock>>>(deviceBitmap);
	cudaMalloc(&prngStates, sizeof(curandState) * threadsPerBlock.x * threadsPerBlock.y);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while allocating memory on the GPU!\n");
		return -1;
	}

	//Setup the random generator states for each thread in a block.
	printf("Initialising the random number generators...\n");
	setupPRNGs<<<numBlocks, threadsPerBlock>>>(prngStates);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while seeding the PRNGs!\n");
		return -1;
	}

	//Send the scene to the card memory.
	printf("Copying the scene to the CUDA device...\n");
	cudaMemcpy(deviceScene.spheres, testScene.spheres, testScene.size * sizeof(Sphere), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error while copying the scene!\n");
		return -1;
	}
	
	//Setup the timing for the actual pathtracing.
	float time;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int noSamples = 10000;

	//Perform noSamples samples and accumulate them in the bitmap.
	printf("Starting the kernels...\n");
	for (int i = 0; i < noSamples; i++) {
		rayTrace<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * sizeof(float3) * 2>>>(deviceScene, deviceBitmap, cameraPos, cameraToImagePlane, xPixel, yPixel, prngStates);
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
	}

	//Stop the CUDA timer.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//Divide each pixel in the bitmap by the number of samples.
	float factor = 1.0f/noSamples;
	scale<<<numBlocks, threadsPerBlock>>>(deviceBitmap, factor);

	printf("Job finished, copying the buffer back...\n");

	//The local bitmap is the same as the device bitmap, except that we need to actually
	//get the floating point values from the device.
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

	unsigned long raysTraced = (long)noSamples * (long)resX * (long)resY * 5;
	printf("Traced %lu rays in %3.1f ms, average speed: %3.1f MRays/s", raysTraced, time, (raysTraced / 1000.0f / time));

	getch();
}