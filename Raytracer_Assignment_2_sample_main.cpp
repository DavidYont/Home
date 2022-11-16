#include "assignment.hpp"

// ******* Function Member Implementation *******

// ***** Shape function members *****
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
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

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
// Might work?
/*
Plane::Plane(atlas::math::Point point, atlas::math::Normal normal) :
    mPoint{point}, mNormal{normal}
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{
    float a = glm::dot(mPoint, mNormal);
    float b = glm::dot(mNormal, ray.o);
    float c = glm::dot(ray.d, mNormal);

    float t = (a - b) / c;
    bool yn{ intersectRay(t) };
    // update ShadeRec info about new closest hit
    if (yn == 1 && t < sr.t)
    {
        sr.normal = mNormal;
        sr.ray = ray;
        sr.color = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }
    return yn;
}

bool Plane::intersectRay(float& tMin) const
{
    //If the ray is parallel to the plane we need to return false, otherwise at some point it hits
    double kEpsilon = 0.0001;

    if (tMin > kEpsilon) {
        return true;
    }
    return false;
}
*/
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

	Vector wo = -sr.ray.o;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t numLights = sr.world->lights.size();

	for (size_t i{ 0 }; i < numLights; ++i)
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
/*
phong::phong() :
    Material{},
    mDiffuseBRDF{ std::make_shared<lambertian>() },
    mAmbientBRDF{ std::make_shared<lambertian>() }
{}

phong::phong(float kd, float ka, Colour color) : phong{}
{
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
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

void phong::setSpecularReflection(float k);
{
    mSpecularBRDF->setDiffuseReflection
}


void phong::setDiffuseColour(Colour colour)
{
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}

*/
// what does w.ambient_ptr represent? 
/**
Colour phong::shade(ShadeRec& sr) {

    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector  wo = wo - sr.ray.d;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr); //Not sure what the translation is for some of these variables
    int num_lights = sr.world->lights.size();

    for (int j{ 0 }; j < num_lights; j++) {

        Vector wi = sr.world->lights[j]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
            L += (mDiffuseBRDF->fn(sr, wo, wi) +
                mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr) * nDotWi;
    }

    return (L);
}
//WIP WIP WIP WIP END
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
{}

atlas::math::Vector ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return atlas::math::Vector{ 0.0f };
}

// ***** Point function members*****
// WIP WIP WIP WIPO WIOPW IPW WIP WIP
point::point() : Light{}
{}


atlas::math::Vector point::getDirection([[maybe_unused]] ShadeRec& sr)
{
    return (mLocation - sr.t);
}

//??????
/*
point::L(const ShadeRec& sr) const {
    return (ls * color);
}
*/
// WIP WIP WIP WIPO WIOPW IPW WIP WIP

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

// ******* OUT OF GAMUT HANDLER *******
Colour
MaxToOne(Colour c) {
    float max_value = glm::max(c.r, glm::max(c.g, c.b));

    if (max_value > 1.0f) {
        return (c / max_value);
    }
    else {
        return c;
    }
}


void Pinhole::renderScene(std::shared_ptr<World>  world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{};

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
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = rayDirection(pixelPoint);

				for (auto const& obj : world->scene)
				{
					obj->hit(ray, trace_data);
				}

				pixelAverage += trace_data.color;
			}

			
            if (pixelAverage.r > 1 || pixelAverage.g > 1 || pixelAverage.b > 1) {
                pixelAverage = MaxToOne(pixelAverage);
            }
           
			world->image.push_back({ pixelAverage.r * avg,
								   pixelAverage.g * avg,
								   pixelAverage.b * avg });
		}
	}
}



// ******* Driver Code *******

int main()
{
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };

    world->width = 600;
    world->height = 600;
    world->background = { 0, 0, 0 };
    world->sampler = std::make_shared<Jittered>(4, 83);

	//Sphere 1
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 0, 0, -600}, 128.0f));
    world->scene[0]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 1, 0, 0 }));
    world->scene[0]->setColour({ 1, 0, 0 });

	//Sphere 2
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 128, 32, -700}, 64.0f));
    world->scene[1]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 0, 0, 1 }));
    world->scene[1]->setColour({ 0, 0, 1 });

	//Sphere 3
    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ -128, 32, -700}, 64.0f));
    world->scene[2]->setMaterial(
        std::make_shared<matte>(0.50f, 0.05f, Colour{ 0, 1, 0 }));
    world->scene[2]->setColour({ 0, 1, 0 });

	//Ground plane
	/*
	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 150, 100, 500 }, atlas::math::Normal { 0, -1, 1 }));
	world->scene[2]->setMaterial(
		std::make_shared<matte>(0.50f, 0.05f, Colour{ 0, 1, 1 }));
	world->scene[2]->setColour({ 0, 1, 0 });
	 */
	world->ambient = std::make_shared<ambient>();
	world->lights.push_back(
		std::make_shared<directional>(directional{ {0, 0, 1024} }));
    world->lights.push_back(
        std::make_shared<point>(point{ {600, 500, 100 } }));
   
	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(0.05f);


	world->lights[0]->setColour({ 1, 1, 1 });
	world->lights[0]->scaleRadiance(4.0f);

	Pinhole camera{};

	camera.setEye({ 150.0f, 150.0f, 500.0f });

	camera.computeUVW();

	camera.renderScene(world);


    saveToBMP("H:/CSC305/Assignments/a2/bundle/bundle/raytrace.bmp", world->width, world->height, world->image);

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
