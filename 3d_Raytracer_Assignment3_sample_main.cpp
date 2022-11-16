#include "assignment.hpp"

// ******* Function Member Implementation *******

// ***** Shape function members *****
int numLights = 0;

Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
    mColour = col;
}

Colour Shape::getColour() const
{
    return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
    mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
    return mMaterial;
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
    mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
    mSamples.reserve(mNumSets* mNumSamples);
    setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
    return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
    mShuffledIndeces.reserve(mNumSamples * mNumSets);
    std::vector<int> indices;

    std::random_device d;
    std::mt19937 generator(d());

    for (int j = 0; j < mNumSamples; ++j)
    {
        indices.push_back(j);
    }

    for (int p = 0; p < mNumSets; ++p)
    {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int j = 0; j < mNumSamples; ++j)
        {
            mShuffledIndeces.push_back(indices[j]);
        }
    }
}

atlas::math::Point Sampler::sampleUnitSquare()
{
    if (mCount % mNumSamples == 0)
    {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

// ***** Camera fucntion members *****

Camera::Camera() :
	mEye{ 0.0f, 0.0f, 500.0f },
	mLookAt{ 0.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
	mUp = up;
}

void Camera::computeUVW()
{
	mW = glm::normalize(mEye - mLookAt);
	mU = glm::normalize(glm::cross(mUp, mW));
	mV = glm::cross(mW, mU);

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y > mLookAt.y)
	{
		mU = { 0.0f, 0.0f, 1.0f };
		mV = { 1.0f, 0.0f, 0.0f };
		mW = { 0.0f, 1.0f, 0.0f };
	}

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
		mEye.y < mLookAt.y)
	{
		mU = { 1.0f, 0.0f, 0.0f };
		mV = { 0.0f, 0.0f, 1.0f };
		mW = { 0.0f, -1.0f, 0.0f };
	}
}

// ***** Light function members *****
Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
	return mRadiance * mColour;
}

void Light::scaleRadiance([[maybe_unused]] float b)
{
	mRadiance = b;
}

void Light::setColour([[maybe_unused]] Colour const& c)
{
	mColour = c;
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
    mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    atlas::math::Vector tmp = ray.o - mCentre;
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.normal = (tmp + t * ray.d) / mRadius;
        sr.ray = ray;
        sr.color = mColour;
		sr.hitPoint = ray.o + t * ray.d;
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

//WIP WIP WIP
// Not using until phong and glossy basic works
/*
bool Sphere::shadow_hit(const Ray& ray, float& tmin) const {
	const auto a = glm::dot(mPoint, mNormal);

	float t = (a - ray.o) * n / (ray.d * n);
	const float kEpsilon = 0.0001f;

	if (t > kEpsilon) {
		tmin = t;
		return (true);
	}
	else
		return (false);
}
*/

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    const auto tmp{ ray.o - mCentre };
    const auto a{ glm::dot(ray.d, ray.d) };
    const auto b{ 2.0f * glm::dot(ray.d, tmp) };
    const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
    const auto disc{ (b * b) - (4.0f * a * c) };

    if (atlas::core::geq(disc, 0.0f))
    {
        const float kEpsilon{ 0.01f };
        const float e{ std::sqrt(disc) };
        const float denom{ 2.0f * a };

        // Look at the negative root first
        float t = (-b - e) / denom;
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }

        // Now the positive root
        t = (-b + e);
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }
    }

    return false;
}
// ***** Plane function members *****

Plane::Plane(atlas::math::Point point, atlas::math::Normal normal) :
    mPoint{point}, mNormal{normal}
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{

	float t{ std::numeric_limits<float>::max() };
 
    bool yn{ intersectRay(ray, t) };
    // update ShadeRec info about new closest hit
    if (yn && t < sr.t)
    {
        sr.normal = mNormal;
		sr.hitPoint = ray.o + t * ray.d;
        sr.ray = ray;
        sr.t = t;
        sr.material = mMaterial;
		sr.color = mColour;
		return true;
    }
    return false;
}

//WIP WIP WIP
// Not using until phong and glossy basic works
/*
bool Plane::shadow_hit(const Ray& ray, float& tmin) const {
	const auto a = glm::dot(mPoint, mNormal);

	float t = (a - ray.o) * n / (ray.d * n);
	const float kEpsilon = 0.0001f;

	if (t > kEpsilon) {
		tmin = t;
		return (true);
	}
	else
		return (false);
}
*/
bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float & tMin) const
{
    //If the ray is parallel to the plane we need to return false, otherwise at some point it hits
    const float kEpsilon = 0.0001f;

	const auto a = glm::dot(mPoint, mNormal);
	const auto b = glm::dot(mNormal, ray.o);
	const auto c = glm::dot(ray.d, mNormal);

	float t = (a - b) / c;

	if (atlas::core::geq(t, kEpsilon))
	{
		tMin = t;
		return true;
	}

    return false;
}

// ***** triangle function members *****

triangle::triangle() :
	Shape(),
	v0(0, 0, 0), v1(0, 0, 1), v2(1, 0, 0),
	normal(0, 1, 0)
{}
triangle::triangle(const atlas::math::Point& a, const atlas::math::Point& b, const atlas::math::Point& c) :
	Shape(),
	v0(a), v1(b), v2(c) {
	normal = glm::cross(atlas::math::Vector(v1 - v0), atlas::math::Vector(v2 - v0));
	glm::normalize(normal);
}

bool triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{

	float t{ std::numeric_limits<float>::max() };

	bool yn{ intersectRay(ray, t) };
	// update ShadeRec info about new closest hit
	if (yn && t < sr.t)
	{
		sr.normal = normalize(normal);
		sr.hitPoint = ray.o + t * ray.d;
		sr.ray = ray;
		sr.t = t;
		sr.material = mMaterial;
		sr.color = mColour;
		return true;
	}
	return false;
}

//WIP WIP WIP
// Not using until phong and glossy basic works
/*
bool triangle::shadow_hit(const Ray& ray, float& tmin) const {
	const auto a = glm::dot(mPoint, mNormal);

	float t = (a - ray.o) * n / (ray.d * n);
	const float kEpsilon = 0.0001f;

	if (t > kEpsilon) {
		tmin = t;
		return (true);
	}
	else
		return (false);
}
*/

//Dont use this!!!!!!!!!!!
/*
bool triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const {
	const float kEpsilon = 0.0001f;

	float D = glm::dot(normal, v0);
	float t = -(glm::dot(normal, ray.o) + D) / dot(normal, ray.d);
	tMin = t;

	atlas::math::Vector P = ray.o + t * ray.d;

	atlas::math::Vector edge0 = (v1 - v0);
	atlas::math::Vector edge1 = (v2 - v1);
	atlas::math::Vector edge2 = (v0 - v2);
	atlas::math::Vector C0 = (P - v0);
	atlas::math::Vector C1 = (P - v1);
	atlas::math::Vector C2 = (P - v2);

	if (glm::dot(normal, glm::cross(edge0, C0)) > 0 &&
		glm::dot(normal, glm::cross(edge1, C1)) > 0 &&
		glm::dot(normal, glm::cross(edge2, C2)) > 0)
	{
		return true;
	}


	return false;


}
*/

bool triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const{

	const float kEpsilon = 0.0001f;

	double a = (double)v0.x - (double)v1.x, b = (double)v0.x - (double)v2.x, c = (double)ray.d.x, d = (double)v0.x - (double)ray.o.x;
	double e = (double)v0.y - (double)v1.y, f = (double)v0.y - (double)v2.y, g = (double)ray.d.y, h = (double)v0.y - (double)ray.o.y;
	double i = (double)v0.z - (double)v1.z, j = (double)v0.z - (double)v2.z, k = (double)ray.d.z, l = (double)v0.z - (double)ray.o.z;

	double m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	double q = g * i - e * k, s = e * j - f * i;

	double inv_denom = 1.0 / (a * m + b * q + c * s);

	double e1 = d * m - b * n - c * p;
	double beta = e1 * inv_denom;

	if (beta < 0.0)
		return (false);

	double r = r = e * l - h * i;
	double e2 = a * n + d * q + c * r;
	double gamma = e2 * inv_denom;

	if (gamma < 0.0)
		return (false);

	if (beta + gamma > 1.0)
		return (false);

	double e3 = a * p - b * r + d * s;
	double t = e3 * inv_denom;

	if (atlas::core::geq(kEpsilon,(float)t))
		return (false);

	tMin = (float)t;
	return (true);
}



// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Regular::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
            }
        }
    }
}

// ***** Random function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Random::generateSamples()
{
    atlas::math::Random<float> engine;
    for (int p = 0; p < mNumSets; ++p)
    {
        for (int q = 0; q < mNumSamples; ++q)
        {
            mSamples.push_back(atlas::math::Point{
                engine.getRandomOne(), engine.getRandomOne(), 0.0f });
        }
    }
}

// ****** Jittered function members ******
Jittered::Jittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Jittered::generateSamples()
{
    atlas::math::Random<float> engine;
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ (q + engine.getRandomOne()) / n, (p + engine.getRandomOne()) / n, 0.0f });
            }
        }
    }
}

// ****** lambertian function members ******
lambertian::lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

lambertian::lambertian(Colour diffuseColor, float diffuseReflection) :
	mDiffuseColour{diffuseColor}, mDiffuseReflection{diffuseReflection}
{}

Colour
lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour
lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return mDiffuseColour * mDiffuseReflection;
}

void lambertian::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void lambertian::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}

// ****** GlossySpecular function members ******
GlossySpecular::GlossySpecular() : mDiffuseColour{}, mDiffuseReflection{}, mSpecularReflection{}
{}

GlossySpecular::GlossySpecular(Colour diffuseColor, float diffuseReflection, float specularReflection) :
	mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }, mSpecularReflection{specularReflection}
{}

Colour GlossySpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	Colour L = mDiffuseColour * mDiffuseColour;

	float nDotWi = glm::dot(sr.normal, incoming);
	atlas::math::Vector r(-incoming.x + 2.0 * sr.normal.x * nDotWi,
		-incoming.y + 2.0 * sr.normal.y * nDotWi,
		- incoming.z + 2.0 * sr.normal.z * nDotWi);
	float rDotWo = glm::dot(r, reflected);


	if (rDotWo > 0.0)
		L = mDiffuseColour * mDiffuseColour * (mSpecularReflection * glm::pow(rDotWo, (float)glm::exp(1))); // This is literally what the textbook says to do and it doesnt work

	return (L);
}

Colour
GlossySpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour(0,0,0); //Return black according to chapter 13
}

void GlossySpecular::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void GlossySpecular::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}

void GlossySpecular::setSpecularReflection(float ks)
{
	mSpecularReflection = ks; //Throwing error
}


// ****** matte function members ******
matte::matte() :
	Material{},
	mDiffuseBRDF{std::make_shared<lambertian>()},
	mAmbientBRDF{std::make_shared<lambertian>()}
{}

matte::matte(float kd, float ka, Colour color) : matte{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setDiffuseColour(color);
}

void matte::setDiffuseReflection(float k)
{
	mDiffuseBRDF->setDiffuseReflection(k);
}

void matte::setAmbientReflection(float k)
{
	mAmbientBRDF->setDiffuseReflection(k);
}

void matte::setDiffuseColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
}

Colour matte::shade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	size_t numberer = sr.world->lights.size();

	for (size_t i{ 0 }; i < numberer; i++)
	{
		Vector wi = sr.world->lights[i]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);

		if (nDotWi > 0.0f)
		{
			L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
		}
	}

	return L;
}

// ***** Phong function members *****
// WIP WIP WIP WIP WIPW IPW IPWIPOW WIOPW IPWIPW

phong::phong() :
    Material{},
    mDiffuseBRDF{ std::make_shared<GlossySpecular>() },
    mAmbientBRDF{ std::make_shared<GlossySpecular>() }
{}

phong::phong(float kd, float ka, float ks, Colour color) : phong{}
{
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
	setSpecularReflection(ks);
    setDiffuseColour(color);
}

void phong::setDiffuseReflection(float k)
{
    mDiffuseBRDF->setDiffuseReflection(k);
}

void phong::setAmbientReflection(float k)
{
    mAmbientBRDF->setDiffuseReflection(k);
}

//WIP
void phong::setSpecularReflection(float k)
{
	mSpecularBRDF->setSpecularReflection(k);
}


void phong::setDiffuseColour(Colour colour)
{
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}


// what does w.ambient_ptr represent? 

Colour phong::shade(ShadeRec& sr) {

    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector  wo = wo - sr.ray.d; //This seems wrong
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr); //Not sure what the translation is for some of these variables
    size_t num_lights = sr.world->lights.size();

    for (int j{ 0 }; j < num_lights; j++) {

        Vector wi = sr.world->lights[j]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
            L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr) * nDotWi;
    }

    return (L);
}

//PHONG FOR SHADOWS USE THIS
/*
Colour phong::shade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector  wo = wo - sr.ray.d; //This seems wrong
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr); //Not sure what the translation is for some of these variables
	int num_lights = sr.world->lights.size();

	for (int j = 0; j < num_lights; j++) {
		Vector wi = sr.world->lights[j]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);

		if (nDotWi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->casts_shadows()) {
				Ray shadowRay(sr.hitPoint, wi);
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow)
				L += (mDiffuseBRDF->fn(sr, wo, wi) +
					mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr) * nDotWi;
		}
	}

	return (L);
}
//WIP WIP WIP WIP END
*/

/************REFLECTIONS**************************/
/*
Colour Whitted::trace_ray(const Ray ray, const int depth) const {
	if (depth > world_ptr->vp.max_depth)
		return (Colour(0,0,0));
	else {
		ShadeRec sr(world_ptr->hit_objects(ray));
		//sr.hit_an_object
		if (sr.t) {
			sr.depth = depth; //?
			sr.ray = ray;

			return (sr.material_ptr->shade(sr));
		}
		else
			return (world_ptr->background_color);
	}
}
*/

// ***** Directional function members *****
directional::directional() : Light{}
{}

directional::directional(atlas::math::Vector const& d) : Light{}
{
	setDirection(d);
}

void directional::setDirection(atlas::math::Vector const& d)
{
	mDirection = glm::normalize(d);
}

atlas::math::Vector directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return mDirection;
}

// ***** Ambient function members*****
ambient::ambient() : Light{}
{
	scaleRadiance(1.0f);
	setColour(Colour{ 1,1,1 });
}

atlas::math::Vector ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return atlas::math::Vector{ 0.0f };

}

// ***** Point function members*****

point::point() : Light{}
{}

point::point(atlas::math::Vector const& l) : Light{}
{
		mLocation = l;
}


atlas::math::Vector point::getDirection(ShadeRec& sr)
{
	
	return (glm::normalize(mLocation - sr.hitPoint));
}

/*
bool point::in_shadow(const Ray& ray, const ShadeRec& sr) const {
	float t;
	int num_objects = sr.world->objects.size();
	float d = location.distance(ray.o);

	for (int j = 0; j < num_objects; j++)
		if (sr.w.objects[j]->shadow_hit(ray, t) && t < d)
			return (true);

	return (false);
}

void AmbientOccluder::set_sampler(Sampler* s_ptr) {
	if (sampler_ptr) {
		delete sampler_ptr;
		sampler_ptr = NULL;
	}

	sampler_ptr = s_ptr;
	sampler_ptr->map_samples_to_hemisphere(1); //Defined after shadows I assume
}

atlas::math::Vector AmbientOccluder::get_direction(ShadeRec& sr) {
	atlas::math::Point sp = sampler_ptr->sample_hemisphere(); //Defined after shadows I assume
	return (sp.x * u + sp.y * v + sp.z * w);
}

bool AmbientOccluder::in_shadow(const Ray& ray, const ShadeRec& sr) const {
	float t;
	int num_objects = sr.world->scene.size();

	for (int j = 0; j < num_objects; j++)
		if (sr.world->scene[j]->shadow_hit(ray, t));
			return (true);

	return (false);
}

Colour AmbientOccluder::L(ShadeRec& sr) {
	w = sr.normal;
	// jitter up vector in case normal is vertical
	v = w ^ atlas::math::Vector(0.0072, 1.0, 0.0034);
	v = glm::normalize(v);
	u = v ^ w;

	atlas::math::Ray shadow_ray;
	shadow_ray.o = sr.hit_point;
	shadow_ray.d = get_direction(sr);

	if (in_shadow(shadow_ray, sr))
		return (min_amount * ls * colour);
	else
		return (ls * colour);
}
*/
// ***** Pinhole function members*****
Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance)
{
	mDistance = distance;
}

void Pinhole::setZoom(float zoom)
{
	mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const
{
	const auto dir = p.x * mU + p.y * mV - mDistance * mW;
	return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World>  world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{};

	float max = 1.0f;

	ray.o = mEye;
	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{

	

		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{

				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = rayDirection(pixelPoint);

				for (auto const& obj : world->scene)
				{
					obj->hit(ray, trace_data);		
				}

				//HERE IS WHERE THE CALCULATIONS WILL HAPPEN, PROBABLY?
				if (trace_data.material != NULL) {
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}
            
			float pixelR = pixelAverage.r * avg;
			float pixelG = pixelAverage.g * avg;
			float pixelB = pixelAverage.b * avg;

			

			if (pixelR > max) {
				max = pixelR;
			}
			if (pixelG > max) {
				max = pixelG;
			}
			if (pixelB > max) {
				max = pixelB;
			}

            
			world->image.push_back({ pixelR,
								   pixelG,
								   pixelB});
		}
	}
	int i{ 0 };
	for (Colour c : world->image) {
		world->image[i] = { c[0] / max, c[1] / max, c[2] / max };
		i++;
	}
}



// ******* Driver Code *******

int main()
{
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };
	

    world->width = 1002;
    world->height = 1002;
    world->background = { 0, 0, 0 };
    world->sampler = std::make_shared<Jittered>(16, 83); // 4,83

	/*************MATTE OBJECTS*************************/
	
	//Sphere 1
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 0, 200, 300}, 128.0f));
    world->scene[0]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 1, 0, 0 }));
    world->scene[0]->setColour({ 1, 0, 0 });

	//Sphere 2
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 128, 100, 400}, 64.0f));
    world->scene[1]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 0, 0, 1 }));
    world->scene[1]->setColour({ 0, 0, 1 });

	//Sphere 3
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 300, 300, 450}, 64.0f));
    world->scene[2]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 0, 1, 0 }));
    world->scene[2]->setColour({ 0, 1, 0 });

	//Ground plane
		world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0, 450, 0 }, atlas::math::Normal { 0, 1, 0 }));
	world->scene[3]->setMaterial(
		std::make_shared<matte>(0.50f, 0.05f, Colour{ .5, .5, .5 }));
	world->scene[3]->setColour({ .5, .5, .5 });

	//Triangle 1
	
	world->scene.push_back(
		std::make_shared<triangle>(atlas::math::Point{ 100, 300, 0 }, atlas::math::Point{ 300, 100, 450 }, atlas::math::Point{ 0, 200, 300 }));
	world->scene[4]->setMaterial(
		std::make_shared<matte>(0.50f, 0.05f, Colour{ 1,1,1 }));
	world->scene[4]->setColour({ 1,1,1 });
	

	
	/***********GLOSSY OBJECTS**************/
	/*
	//Sphere 1
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, 200, 300 }, 128.0f));
	world->scene[0]->setMaterial(
		std::make_shared<phong>(0.50f, 0.05f, 0.1f, Colour{ 1, 0, 0 }));
	world->scene[0]->setColour({ 1, 0, 0 });

	//Sphere 2
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 128, 100, 400 }, 64.0f));
	world->scene[1]->setMaterial(
		std::make_shared<phong>(0.50f, 0.05f, 0.05f, Colour{ 0, 0, 1 }));
	world->scene[1]->setColour({ 0, 0, 1 });

	//Sphere 3
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 300, 300, 450 }, 64.0f));
	world->scene[2]->setMaterial(
		std::make_shared<phong>(0.50f, 0.05f, 0.05f, Colour{ 0, 1, 0 }));
	world->scene[2]->setColour({ 0, 1, 0 });

	//Ground plane
	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0, 450, 0 }, atlas::math::Normal{ 0, 1, 0 }));
	world->scene[3]->setMaterial(
		std::make_shared<phong>(0.50f, 0.05f, 0.05f, Colour{ .5, .5, .5 }));
	world->scene[3]->setColour({ .5, .5, .5 });

	//Triangle 1

	world->scene.push_back(
		std::make_shared<triangle>(atlas::math::Point{ 100, 300, 0 }, atlas::math::Point{ 300, 100, 450 }, atlas::math::Point{ 0, 200, 300 }));
	world->scene[4]->setMaterial(
		std::make_shared<phong>(0.50f, 0.05f, 0.05f, Colour{ 1,1,1 }));
	world->scene[4]->setColour({ 1,1,1 });
	
	*/
	/*****************LIGHTS*****************/
	
	world->ambient = std::make_shared<ambient>();
	world->lights.push_back(
		std::make_shared<point>(point{ { 0, -100, 0 } }));
	//world->lights.push_back(
		//std::make_shared<directional>(directional{ {0, -1, 0} }));
   
	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(1.0f);


	world->lights[0]->setColour({ 1, 1, 1 });
	world->lights[0]->scaleRadiance(15.0f);
	numLights++;

	
	//world->lights[1]->setColour({ 1, 1, 1 });
	//world->lights[1]->scaleRadiance(5.0f);
	//numLights++;
	

	Pinhole camera{};

	camera.setEye({ 0.0f, 50.0f, -900.0f });

	camera.computeUVW();

	camera.renderScene(world);


    saveToBMP("C:/CSC305/Assignment3/bundle/bundle/raytrace.bmp", world->width, world->height, world->image);

    return 0;
}


/**
 * Saves a BMP image file based on the given array of pixels. All pixel values
 * have to be in the range [0, 1].
 *
 * @param filename The name of the file to save to.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param image The array of pixels representing the image.
 */
void saveToBMP(std::string const& filename,
               std::size_t width,
               std::size_t height,
               std::vector<Colour> const& image)
{
    std::vector<unsigned char> data(image.size() * 3);

    for (std::size_t i{0}, k{0}; i < image.size(); ++i, k += 3)
    {
        Colour pixel = image[i];
        data[k + 0]  = static_cast<unsigned char>(pixel.r * 255);
        data[k + 1]  = static_cast<unsigned char>(pixel.g * 255);
        data[k + 2]  = static_cast<unsigned char>(pixel.b * 255);
    }

    stbi_write_bmp(filename.c_str(),
                   static_cast<int>(width),
                   static_cast<int>(height),
                   3,
                   data.data());
}
