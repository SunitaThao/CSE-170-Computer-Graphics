#include "framework.h"
#include <iostream>

//Summary: renders image of created scene and loads it in as a texture

enum MaterialType { ROUGH, REFLECTIVE, REFRACTIVE };
bool key_states[256];
int a = 1;

struct Material { //includes parameters needed for illumination models
	vec3 ka, kd, ks; //ambient diffuse specular reflectiviity
	float  shininess; //shininess
	vec3 F0; //surface rep for pependic illumination
	float ior; //transparent = refractive, index of refraction is scalar, rgb in same direction, measures bending of a ray of light
	MaterialType type; //store type of material, rough,reflective, transparent or refractive
	Material(MaterialType t){ 
		type = t;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) { //sets material type to rough
		ka = _kd * M_PI; //ambient
		kd = _kd; //diffuse
		ks = _ks; //specular
		shininess = _shininess; //initialize shininess
	}
};

vec3 operator/(vec3 num, vec3 denom) { // just helps do the division math operation for each x y and z variable
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material { //index of refraction(measure of bending of a light ray) and extinction coeeficient(how strongly something absorbs or reflects light) parameter which is different for r, g, and b = kappa
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) { //sets material type to reflective
		vec3 ones(1, 1, 1);
		F0 = ((n - ones)*(n - ones) + kappa * kappa) / ((n + ones)*(n + ones) + kappa * kappa); //final function for perpendicular illumination, all vec3
	}
};

struct RefractiveMaterial : Material { //transparent material, kappa(extinction coefficient) must be zero for object to be transparent
	RefractiveMaterial(vec3 n) : Material(REFRACTIVE) {//index of refraction is the only parameter included
		vec3 ones(1, 1, 1);
		F0 = ((n - ones)*(n - ones)) / ((n + ones)*(n + ones)); //no kappa
		ior = n.x;//index of refraction can store ior avg of 3 wavelengths, but we only did it for the red component here
	}
};

struct Hit {//computes ior
	float h;//ray parameter of intersection, a hough line, must be positive for valid intersection
	vec3 position, normal;//location of intersection
	Material * material; //material at intersection point
	Hit(){ 
		h = -1;//example of no intersection
	}
};

struct Ray { //hough line idenitifed by start and dir
	vec3 start, direction;
	Ray(vec3 _start, vec3 _dir){
		start = _start; direction = normalize(_dir);//normalize direction with assumption of unit vectors
	}
};

class Intersectable { //base class, impt to compute intersection between surface and ray,
	//abstract because if we dont know equation cant really compute interesction
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;//intersect function does the intersection calculation
};

class Sphere : public Intersectable {//have to derive from Intersectable 
	vec3 center;
	float radius;
public:
	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center; radius = _radius; material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float a = dot(ray.direction, ray.direction);
		float b = dot(dist, ray.direction) * 2.0f;
		float c = dot(dist, dist) - radius * radius;

		//inserting ray equ intoimplicit equ of sphere to get discriminant
		float discr = b * b - 4.0f * a * c; //discriminant of 2nd order eq, defines if there is an intersection or not
		if (discr < 0) return hit;//returns hit indicating a neg number (-1) initialized in hit constructor 
		//else we solve it and get two roots
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		if (t1 <= 0) return hit;//checks root to be pos or neg

		hit.h = (t2 > 0) ? t2 : t1; //need root that is positive and if have 2 positive roots, we take the smaller number

		//if we have valid intersection
		hit.position = ray.start + ray.direction * hit.h; //hit parameter will be included into ray equ 
		hit.normal = (hit.position - center) / radius; //normal vector of surface at intersection point, for a sphere the difference between hit position and center is always perpendicular to surface
		//in doing so, we'll get radius/radius resulting in 1
		hit.material = material;

		return hit;
	}
};

class Camera {//rep user eye position
	vec3 eye, look, right, up;//look rep center of rectangle physcial screen, right is a vector from center to right edge of screen
	//up is vector from center to top of screen
	float fov;
public:
	void set(vec3 _eye, vec3 _look, vec3 vup, float _fov) {//makes sure diff between eye and look are perpendicular to each other so we dont get a distorted img
		eye = _eye; look = _look; fov = _fov;
		vec3 d = eye - look;
		float windowSize = length(d) * tanf(fov / 2);
		right = normalize(cross(vup, d)) * windowSize;
		up = normalize(cross(d, right)) * windowSize;
	}

	Ray getRay(int x, int y) {//computes a ray corrspeonding to a physical pixel of pixel coord x and y that are first converted to normalized coord for x and y, in -1 to +1 interval
		vec3 direction = look + right * (2 * (x + 0.5f) / windowWidth - 1) + up * (2 * (y + 0.5f) / windowHeight - 1) - eye;//a point on camera rect - eye to get direction of ray
		return Ray(eye, direction);
	}

	void Animate(float anim) { //camera animation rotates eye around camera point, view scene from different perspective
		vec3 e = eye - look;
		eye = vec3(e.x * cos(anim) + e.z * sin(anim), e.y, -e.x * sin(anim) + e.z * cos(anim)) + look;
		set(eye, look, up, fov);
	}
};

struct Light {//handle direction of light sources
	vec3 direction;
	vec3 Le; //emission intensity is a spectrum so has different wavelength for r,g,b so it has its own variable
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction); //normalize because in illumation computation, vectors are unit vectors
		Le = _Le;
	}
};

const float epsilon = 0.0001f; // used to help fix that small error when the ray actually hits the surface so that it doesn't count as completely just 0

class Scene {
	std::vector<Intersectable *> objects; //scene is heterogenous collection of objects, cause can derive different geometric types 
	std::vector<Light *> lights;//collection of light sources that are arbitrary
	Camera camera;//single camera
	vec3 La;//ambient light available in all points in all directions
public:
	void build(int a) {//bulds up virtual world, objects, light sources, initializes camera and ambient light source
		if (a == 1) {
			vec3 eye = vec3(0, 0, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);//specifiy location of eye, vertical dir and look point for camera
			float fov = 45 * M_PI / 180;
			camera.set(eye, lookat, vup, fov);//computes parameter needed in camera 

			La = vec3(0.4f, 0.4f, 0.4f);//ambient light intensity of rgb
			vec3 lightDirection(1, 1, 1), Le(2, 2, 2); //initial single light source direction with that dir vector and emission intensity
			lights.push_back(new Light(lightDirection, Le));
			
			vec3 ks(2, 2, 2);//specular reflectivity initialized
			//definition of geometric object, all spheres, center radius material property, use same specular reflectivity for all rough material, different shininess parameter, different diffuse reflectivity , vec3's define colors
			//trasnparent refractive object, 1.5 1.5 1.5 iof, phsycially loss material
			objects.push_back(new Sphere(vec3(0.25, 0.25, 0.75), 0.75, //center & radius
				new RoughMaterial(vec3(0, 0, 0), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, -6.5, 0), 6,
				new ReflectiveMaterial(vec3(0.1, 0.0, 0.0), vec3(3, 2, 1)))); //ior for rgb and extinction reflection for rgb
		}
		else if (a == 2) {
			vec3 eye = vec3(0, 0, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);//specifiy location of eye, vertical dir and lookat point
			float fov = 45 * M_PI / 180;
			camera.set(eye, lookat, vup, fov);//computes parameter needed in camera 

			La = vec3(0.4f, 0.4f, 0.4f);//ambient light intensity of rgb
			vec3 lightDirection(1, 1, 1), Le(2, 2, 2); //initial light direction with that vector and emition intensity
			lights.push_back(new Light(lightDirection, Le));

			vec3 ks(2, 2, 2);
			//definition of geometric object, all spheres, center radius material property, use same specular reflectivity for all rough material, different shininess parameter, different diffuse reflectivity , vec3's define colors
			//trasnparent refractive object, 1.5 1.5 1.5 iof, phsycially loss material
			objects.push_back(new Sphere(vec3(-1.5, 0, 1.5), 0.5,
				new RoughMaterial(vec3(1, 0, 0), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(1.55, 0, 0), 0.5,
				new RoughMaterial(vec3(0, 1, 1), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, 0, 0), 0.5,
				new RefractiveMaterial(vec3(1.2, 1.2, 1.2))));//ior for transparent sphere
			objects.push_back(new Sphere(vec3(0, -6.5, 0), 6,
				new ReflectiveMaterial(vec3(1, 0.5, 1.5), vec3(1, 1, 3)))); //ior for rgb and extinction reflection for rgb
		}
		else if (a == 3) {
			vec3 eye = vec3(0, 0, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);//specifiy location of eye, vertical dir and lookat point
			float fov = 45 * M_PI / 180;
			camera.set(eye, lookat, vup, fov);//computes parameter needed in camera 

			La = vec3(0.4f, 0.4f, 0.4f);//ambient light intensity of rgb
			vec3 lightDirection(1, 1, 1), Le(2, 2, 2); //initial light direction with that vector and emition intensity
			lights.push_back(new Light(lightDirection, Le));

			vec3 ks(2, 2, 2);
			//definition of geometric object, all spheres, center radius material property, use same specular reflectivity for all rough material, different shininess parameter, different diffuse reflectivity , vec3's define colors
			//trasnparent refractive object, 1.5 1.5 1.5 iof, phsycially loss material
			objects.push_back(new Sphere(vec3(-1.3, 0.5, 0), 0.25,
				new RoughMaterial(vec3(1, 1, 0), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(1.7, 0, 0.7), 0.5,
				new RoughMaterial(vec3(1, 0, 1), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, 0.55, 2), 0.75,
				new RoughMaterial(vec3(1, 1, 1), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, 0, 0.6), 0.37,
				new RefractiveMaterial(vec3(1.2, 1.2, 1.2))));//ior for transparent sphere
			objects.push_back(new Sphere(vec3(0, -6.5, 0), 6,
				new ReflectiveMaterial(vec3(0, 0.5, 1), vec3(3, 2, 1)))); //ior for rgb and extinction reflection for rgb
		}
		else {
			vec3 eye = vec3(0, 0, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);//specifiy location of eye, vertical dir and lookat point
			float fov = 45 * M_PI / 180;
			camera.set(eye, lookat, vup, fov);//computes parameter needed in camera 

			La = vec3(0.4f, 0.4f, 0.4f);//ambient light intensity of rgb
			vec3 lightDirection(1, 1, 1), Le(2, 2, 2); //initial light direction with that vector and emition intensity
			lights.push_back(new Light(lightDirection, Le));

			vec3 ks(2, 2, 2);
			//definition of geometric object, all spheres, center radius material property, use same specular reflectivity for all rough material, different shininess parameter, different diffuse reflectivity , vec3's define colors
			//trasnparent refractive object, 1.5 1.5 1.5 iof, phsycially loss material
			objects.push_back(new Sphere(vec3(-0.7, 0, 0), 0.1,
				new RoughMaterial(vec3(0.5, 0.5, 0.1), ks, 50))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(-0.3, 0, 0), 0.2,
				new RoughMaterial(vec3(0.1, 0.5, 0.5), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, 0.5, -0.8), 0.3,
				new RoughMaterial(vec3(0.3, 0.8, 0.6), ks, 20))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0.7, 0, 0), 0.4,
				new RoughMaterial(vec3(0.1, 0.7, 0.1), ks, 50))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0.3, 1, 0), 0.5,
				new RoughMaterial(vec3(0.1, 0.2, 0.9), ks, 100))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(0, 0.5, -1), 0.6,
				new RoughMaterial(vec3(0.5, 0.1, 0.2), ks, 20))); //diffuse, specular, shininess
			objects.push_back(new Sphere(vec3(10, -2.5, 0), 6,
				new ReflectiveMaterial(vec3(1, 0, 3), vec3(5, 9, 8)))); //ior for rgb and extinction reflection for rgb
		}
	}



	void render(std::vector<vec4>& image) {//render computes result into image array, image is computed and available in cpu memory
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {//going through every physical pixel one by one
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y)); //obtain a ray corresponding to pixel in virtual world, trace function will calculate surface first hit by the ray then calculate radiance is reflected by surface in direction of eye(reflected radiance of surface first hit by the ray)
				//written into color var
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);//written into image array
			}
		}

	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.h > 0 && (bestHit.h < 0 || hit.h < bestHit.h))  bestHit = hit; //will have hit info for first intersection in besthit, checks which hit is closer and takes it as best hit
		}
		if (dot(ray.direction, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);//assume normal vector points towards arriving ray for reflective&refractive direction, checks if ray and normal direction are pointing in the right dir and if not then normal vector is fixed
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights, determines if there are any object between shaded point and light source then return true, simpler form of firstIntersect function
		for (Intersectable * object : objects) if (object->intersect(ray).h > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {//recursion depth to limit recursion and prevent crash for stack overflow
		if (depth > 5) return La; //if true, terminate processs and return ambient light else
		Hit hit = firstIntersect(ray);
		if (hit.h < 0) return La; //neg then ray didnt intersect object, only see ambient light, else 

		if (hit.material->type == ROUGH) {
			vec3 outRadiance = hit.material->ka * La;//reflective radiance initialized to ambient reflection 
			for (Light * light : lights) {//visit abstract light source one by one, determine if its visible or not, if yes then added to reflective radiance
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);//used to calculate visibility of light source
				float cosTheta = dot(hit.normal, light->direction); //surface normal and illumination light dir, unit vectors, determines cosine of their angle
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation, if cos < 0 then > 90 degreses then illuminates back side, not pos then ignore calculations, if pos then there is no other obj between light source and obj
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta; //added to calculation
					vec3 halfway = normalize(-ray.direction + light->direction);//3 lines add glossy relfection, first line has view dir and illumination dir(unit vectors), added gets vector in betweeen them
					float cosDelta = dot(hit.normal, halfway);//computes cos angle
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess); //added to calcualtions, powf = power of 
				}
			}//if all lightsource contribution added then return result
			return outRadiance;
		}
		//handles optically smooth surfaces
		float cosa = -dot(ray.direction, hit.normal);//cos of angle between view direction and normal 
		vec3 one(1, 1, 1);
		vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);//final function for perpendicular illumination that uses cosine angle computed
		vec3 reflectedDir = ray.direction - hit.normal * dot(hit.normal, ray.direction) * 2.0f;//to get incident radiance first calculate reflection direction, ideal reflection direction
		vec3 outRadiance = trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;//establish a reflection ray, trace is called recursively and increases depth parameter to help control dephts of recursion and avoid stack overflow
		//get outer radiance variable with ideal reflection direction 

		if (hit.material->type == REFRACTIVE) {//if transparent, add contributiuon of ideal refraction direction
			float disc = 1 - (1 - cosa * cosa) / hit.material->ior / hit.material->ior; // scalar n, formula
			if (disc >= 0) {
				vec3 refractedDir = ray.direction / hit.material->ior + hit.normal * (cosa / hit.material->ior - sqrt(disc));
				outRadiance = outRadiance +
					trace(Ray(hit.position - hit.normal * epsilon, refractedDir), depth + 1) * (one - F);//trace delivers radiance form refraction direction 
			}
		}
		return outRadiance;
	}

	void Animate(float dt) { camera.Animate(dt); }
};
//consider image as a full texture to display it on screen for users, so use shaders
Scene scene1;
Scene scene2;
Scene scene3;
Scene scene0;
GPUProgram gpuProgram; // vertex and fragment shaders

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord; //interpolated 

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL, for every pixel the fragment shader is called
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation, texture quad

	void main() { fragmentColor = texture(textureUnit, texcoord); }
)";

class FullScreenTexturedQuad {//texture quad
	unsigned int vao = 0, textureId = 0;	// vertex array object id and texture id
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad, directly normalized
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);//uniform variable is copied to frag shader
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization(int a) {
	glViewport(0, 0, windowWidth, windowHeight);

	if (a == 0)
		scene0.build(a);
	else if (a == 1)
		scene1.build(a);
	else if (a == 2)
		scene2.build(a);
	else
		scene3.build(a);
	
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor"); 	// create program for the GPU
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);

	if (a == 0)
		scene0.render(image); 						// Execute ray casting
	else if (a == 1)
		scene1.render(image); 						// Execute ray casting
	else if (a == 2)
		scene2.render(image); 						// Execute ray casting
	else
		scene3.render(image); 						// Execute ray casting

	fullScreenTexturedQuad->LoadTexture(image); // copy image to GPU as a texture
	fullScreenTexturedQuad->Draw();				// Display rendered image on screen
	glutSwapBuffers();						// exchange the two buffers
}

void keyboard_func(unsigned char key, int x, int y)
{
	key_states[key] = true;
	switch (key)
	{
		case 'v':
		{
			if (a == 3)
				a = 0;
			else
				a = a + 1;

			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clean the screen and the depth buffer
			glLoadIdentity();
			onInitialization(a);
			onDisplay();

			break;
		}
		// Exit on escape key press
		case '\x1B':
		{
			exit(EXIT_SUCCESS);
			break;
		}
	}
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (a == 0)
		scene0.Animate(0.15f);
	else if (a == 1)
		scene1.Animate(0.2f);
	else if (a == 2)
		scene2.Animate(0.2f);
	else
		scene3.Animate(0.2f);

	glutKeyboardFunc(keyboard_func);
	glutPostRedisplay();
}