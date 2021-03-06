////////////////////////////////////////////////////////////////////////
// Source file for RGBDImage class
////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

#include "RGBD.h"



////////////////////////////////////////////////////////////////////////
// Namespace
////////////////////////////////////////////////////////////////////////

namespace gaps {



////////////////////////////////////////////////////////////////////////
// Constructors/destructors
////////////////////////////////////////////////////////////////////////

RGBDImage::
RGBDImage(void)
  : configuration(NULL),
    configuration_index(-1),
    channels(),
    width(0), height(0),
    camera_to_world(R4Matrix(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1), 0),
    intrinsics(1,0,0, 0,1,0, 0,0,1),
    world_bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
    timestamp(0),
    opengl_texture_id(-1),
    name(NULL),
    color_filename(NULL),
    depth_filename(NULL),
    category_filename(NULL),
    instance_filename(NULL),
    color_resident_count(0),
    depth_resident_count(0),
    category_resident_count(0),
    instance_resident_count(0),
    data(NULL)
{
}



RGBDImage::
RGBDImage(const char *color_filename, const char *depth_filename,
  const R3Matrix& intrinsics_matrix, const R4Matrix& camera_to_world_matrix,
  int width, int height, RNScalar timestamp)
  : configuration(NULL),
    configuration_index(-1),
    channels(),
    width(width), height(height),
    camera_to_world(camera_to_world_matrix, 0),
    intrinsics(intrinsics_matrix),
    world_bbox(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX),
    timestamp(timestamp),
    opengl_texture_id(-1),
    name(NULL),
    color_filename(NULL),
    depth_filename(NULL),
    category_filename(NULL),
    instance_filename(NULL),
    color_resident_count(0),
    depth_resident_count(0),
    category_resident_count(0),
    instance_resident_count(0),
    data(NULL)
{
  // Set filenames
  if (depth_filename) SetDepthFilename(depth_filename);
  if (color_filename) SetColorFilename(color_filename);
}



RGBDImage::
~RGBDImage(void)
{
  // Delete opengl texture
  if (opengl_texture_id >= 0) {
    GLuint i = opengl_texture_id;
    glDeleteTextures(1, &i);
  }
  
  // Remove from configuration
  if (configuration) {
    configuration->RemoveImage(this);
  }

  // Delete channels
  for (int i = 0; i < channels.NEntries(); i++) {
    if (channels[i]) delete channels[i];
  }

  // Delete filenames
  if (name) free(name);
  if (color_filename) free(color_filename);
  if (depth_filename) free(depth_filename);
  if (category_filename) free(category_filename);
  if (instance_filename) free(instance_filename);
}



////////////////////////////////////////////////////////////////////////
// Image property access functions
////////////////////////////////////////////////////////////////////////

R4Matrix RGBDImage::
ModelViewMatrix(void) const
{
  // Return transformation from world coordinates to camera coordinates
  return camera_to_world.Matrix().Inverse();
}



R4Matrix RGBDImage::
ProjectionMatrix(RNScalar neardist, RNScalar fardist) const
{
  // Return projection transformation from camera coordinates to normalized image coordinates
  RNScalar fx = Intrinsics()[0][0];
  RNScalar fy = Intrinsics()[1][1];
  RNScalar cx = Intrinsics()[0][2];
  RNScalar cy = Intrinsics()[1][2];
  RNScalar W = NPixels(RN_X);
  RNScalar H = NPixels(RN_Y);
  return R4Matrix(
    2*fx/W, 0, -2*(cx/W)+1, 0,
    0, 2*fy/H, -2*(cy/H)+1, 0,
    0, 0, -(fardist + neardist) / (fardist - neardist),  -2.0 * neardist * fardist / (fardist - neardist),
    0, 0, -1, 0 );
}



R3Box RGBDImage::
WorldBBox(void) const
{
  // Return remembered bounding box of points in world coordinates, if valid
  if (!world_bbox.IsEmpty()) return world_bbox;

  // Compute world bounding box
  R3Box b = R3null_box;
  if (DepthChannel()) {
    // Compute world bounding box of points
    int skip = 1;
    b = R3null_box;
    for (int iy = 0; iy < height; iy += skip) {
      for (int ix = 0; ix < width; ix += skip) {
        R3Point world_position;
        R2Point image_position(ix+0.5, iy+0.5);
        if (RGBDTransformImageToWorld(image_position, world_position, this)) {
          b.Union(world_position);
        }
      }
    }

    // Just so that don't compute again if empty
    if (b.IsEmpty()) b.Union(WorldViewpoint());

    // Remember world bounding box
    ((RGBDImage *) this)->world_bbox = b;
  }
  else {
    // Return conservative bounding box without reading image
    RNLength max_depth = 7.0;
    RNScalar xfocal = Intrinsics()[0][0];
    RNScalar yfocal = Intrinsics()[1][1];
    RNScalar tan_xfov = (xfocal > 0) ? 0.5*NPixels(RN_X) / xfocal : 1.0;
    RNScalar tan_yfov = (yfocal > 0) ? 0.5*NPixels(RN_Y) / yfocal : 1.0;
    R3Point c = WorldViewpoint() + max_depth * WorldTowards();
    R3Vector dx = WorldRight() * max_depth * tan_xfov * WorldRight();
    R3Vector dy = WorldRight() * max_depth * tan_yfov * WorldUp();
    b.Union(WorldViewpoint());
    b.Union(c - dx - dy);
    b.Union(c + dx - dy);
    b.Union(c - dx + dy);
    b.Union(c + dx + dy);
  }

  // Return computed bounding box
  return b;
}



////////////////////////////////////////////////////////////////////////
// Pixel property access functions
////////////////////////////////////////////////////////////////////////

RNRgb RGBDImage::
PixelColor(int ix, int iy) const
{
  // Return color of pixel
  if (NChannels() < 3) return RNblack_rgb;
  RNScalar r = PixelChannelValue(ix, iy, RGBD_RED_CHANNEL);
  RNScalar g = PixelChannelValue(ix, iy, RGBD_GREEN_CHANNEL);
  RNScalar b = PixelChannelValue(ix, iy, RGBD_BLUE_CHANNEL);
  return RNRgb(r, g, b);
}



R3Vector RGBDImage::
PixelWorldNormal(int ix, int iy) const
{
  // Allocate buffer of points
  const int r = 2;
  R3Point *points = new R3Point[(2*r+1) * (2*r+1)];
  int npoints = 0;
  
  // Fill buffer of points in neighborhood r of (ix, iy)
  for (int s = -r; s <= r; s++) {
    int i = ix + s;
    if ((i < 0) || (i >= width)) continue;
    for (int t = -r; t <= r; t++) {
      int j = iy + t;
      if ((j < 0) || (j >= height)) continue;
      R3Point camera_position;
      R2Point image_position(i + 0.5, j + 0.5);
      if (RGBDTransformImageToCamera(image_position, camera_position, this)) {
        points[npoints++] = camera_position;
      }
    }
  }

  // Compute normal at pixel
  if (npoints < 3) return R3zero_vector;
  R3Point centroid = R3Centroid(npoints, points);
  R3Triad triad = R3PrincipleAxes(centroid, npoints, points);
  R3Vector normal = (triad[2].Z() > 0) ? triad[2] : -triad[2];

  // Transform into world coordinates
  normal.Transform(camera_to_world);
  normal.Normalize();
  
  // Delete buffer of points
  delete [] points;

  // Return normal in world coordinates
  return normal;
}



////////////////////////////////////////////////////////////////////////
// Point propery access functions
////////////////////////////////////////////////////////////////////////

RNRgb RGBDImage::
PixelColor(const R2Point& image_position) const
{
  // Return color of pixel
  if (NChannels() < 3) return RNblack_rgb;
  RNScalar r = PixelChannelValue(image_position, RGBD_RED_CHANNEL);
  RNScalar g = PixelChannelValue(image_position, RGBD_GREEN_CHANNEL);
  RNScalar b = PixelChannelValue(image_position, RGBD_BLUE_CHANNEL);
  return RNRgb(r, g, b);
}



R3Point RGBDImage::
PixelWorldPosition(const R2Point& image_position) const
{
  // Return position of pixel in world coordinates
  R3Point world_position(0,0,0);
  RGBDTransformImageToWorld(image_position, world_position, this);
  return world_position;
}



R3Ray RGBDImage::
PixelWorldRay(const R2Point& image_position) const
{
  // Get/check intrinsics matrix
  const R3Matrix& intrinsics_matrix = Intrinsics();
  if (RNIsZero(intrinsics_matrix[0][0])) return R3null_ray;
  if (RNIsZero(intrinsics_matrix[1][1])) return R3null_ray;

  // Get point at depth 1.0 along ray in camera coordinates
  R3Point camera_position;
  camera_position[0] = (image_position[0] - intrinsics_matrix[0][2]) / intrinsics_matrix[0][0];
  camera_position[1] = (image_position[1] - intrinsics_matrix[1][2]) / intrinsics_matrix[1][1];
  camera_position[2] = -1.0;

  // Create ray and transform into world coordinates
  R3Ray ray(R3zero_point, camera_position);
  ray.Transform(camera_to_world);

  // Return ray through pixel in world coordinates
  return ray;
}



////////////////////////////////////////////////////////////////////////
// Manipulation functions
////////////////////////////////////////////////////////////////////////

void RGBDImage::
SetNPixels(int nx, int ny)
{
  // Resample color channels
  for (int i = RGBD_RED_CHANNEL; i <= RGBD_BLUE_CHANNEL; i++) {
    if (channels.NEntries() <= i) continue;
    if (channels[i]) channels[i]->Resample(nx, ny);
  }

  // Resample other channels (without interpolation)
  for (int i = RGBD_DEPTH_CHANNEL; i < channels.NEntries(); i++) {
    if (channels.NEntries() <= i) continue;
    R3Matrix tmp(R3identity_matrix);
    if (channels[i]) RGBDResampleDepthImage(*(channels[i]), tmp, nx, ny);
  }

  // Update intrinsics 
  double xscale = (nx > 0) ? (double) width / (double) nx : 1;
  double yscale = (ny > 0) ? (double) height / (double) ny : 1;
  this->intrinsics[0][0] /= xscale;
  this->intrinsics[0][2] /= xscale;
  this->intrinsics[1][1] /= yscale;
  this->intrinsics[1][2] /= yscale;

  // Update width and height of image
  this->width = nx;
  this->height = ny;

  // Invalidate opengl
  InvalidateOpenGL();
}



void RGBDImage::
SetPixelColor(int ix, int iy, const RNRgb& color)
{
  // Set pixel red, green, and blue
  SetPixelChannelValue(ix, iy, RGBD_RED_CHANNEL, color.R());
  SetPixelChannelValue(ix, iy, RGBD_GREEN_CHANNEL, color.G());
  SetPixelChannelValue(ix, iy, RGBD_BLUE_CHANNEL, color.B());

  // Invalidate opengl
  InvalidateOpenGL();
}



void RGBDImage::
SetPixelDepth(int ix, int iy, RNScalar depth)
{
  // Set pixel depth
  SetPixelChannelValue(ix, iy, RGBD_DEPTH_CHANNEL, depth);

  // Update bounding box
  R3Point world_position;
  R2Point image_position(ix+0.5, iy+0.5);
  if (RGBDTransformImageToWorld(image_position, world_position, this)) {
    world_bbox.Union(world_position);
  }
}



void RGBDImage::
SetPixelCategory(int ix, int iy, RNScalar category)
{
  // Set pixel depth
  SetPixelChannelValue(ix, iy, RGBD_CATEGORY_CHANNEL, category);
}



void RGBDImage::
SetPixelInstance(int ix, int iy, RNScalar instance)
{
  // Set pixel depth
  SetPixelChannelValue(ix, iy, RGBD_INSTANCE_CHANNEL, instance);
}



void RGBDImage::
SetPixelChannelValue(int ix, int iy, int channel_index, RNScalar value)
{
  // Check channel
  if (!channels[channel_index]) {
    RNFail("RGBD channel is not resident in memory -- cannot set value\n");
    return;
  }
  
  // Set channel value at pixel
  if ((channel_index < 0) || (channel_index >= channels.NEntries())) return;
  if ((ix < 0) || (ix >= channels[channel_index]->XResolution())) return;
  if ((iy < 0) || (iy >= channels[channel_index]->YResolution())) return;
  channels[channel_index]->SetGridValue(ix, iy, value);

  // Update bounding box
  if (channel_index == RGBD_DEPTH_CHANNEL) {
    R3Point world_position;
    R2Point image_position(ix+0.5, iy+0.5);
    if (RGBDTransformImageToWorld(image_position, world_position, this)) {
      world_bbox.Union(world_position);
    }
  }

  // Invalidate opengl
  if ((channel_index >= RGBD_RED_CHANNEL) && (channel_index <= RGBD_BLUE_CHANNEL)) InvalidateOpenGL();
}



void RGBDImage::
SetChannel(int channel_index, const R2Grid& image)
{
  // Check channel
  if (!channels[channel_index]) {
    RNFail("RGBD channel is not resident in memory -- cannot set\n");
    return;
  }
  
  // Copy channel
  *(channels[channel_index]) = image;
  
  // Set width and height
  this->width = image.XResolution();
  this->height = image.YResolution();

  // Invalidate bounding box
  if (channel_index == RGBD_DEPTH_CHANNEL) InvalidateWorldBBox();

  // Invalidate opengl
  if ((channel_index >= RGBD_RED_CHANNEL) && (channel_index <= RGBD_BLUE_CHANNEL)) InvalidateOpenGL();
}



void RGBDImage::
SetColorChannels(const R2Image& image)
{
  // Check if color channels are resident
  if (color_resident_count == 0) {
    RNFail("Unable to set color channels -- they have not been created\n");
    return;
  }

  // Copy color values
  for (int iy = 0; iy < image.Height(); iy++) {
    for (int ix = 0; ix < image.Width(); ix++) {
      RNRgb color = image.PixelRGB(ix, iy);
      channels[RGBD_RED_CHANNEL]->SetGridValue(ix, iy, color.R());      
      channels[RGBD_GREEN_CHANNEL]->SetGridValue(ix, iy, color.G());      
      channels[RGBD_BLUE_CHANNEL]->SetGridValue(ix, iy, color.B());
    }
  }

  // Set width and height
  this->width = image.Width();
  this->height = image.Height();

  // Invalidate opengl
  InvalidateOpenGL();
}



void RGBDImage::
SetDepthChannel(const R2Grid& image)
{
  // Check if color channels are resident
  if (depth_resident_count == 0) {
    RNFail("Unable to set depth channel -- it has not been created\n");
    return;
  }

  // Set depth channel
  SetChannel(RGBD_DEPTH_CHANNEL, image);
}



void RGBDImage::
SetCategoryChannel(const R2Grid& image)
{
  // Check if color channels are resident
  if (category_resident_count == 0) {
    RNFail("Unable to set category channel -- it has not been created\n");
    return;
  }

  // Set category channel
  SetChannel(RGBD_CATEGORY_CHANNEL, image);
}



void RGBDImage::
SetInstanceChannel(const R2Grid& image)
{
  // Check if color channels are resident
  if (instance_resident_count == 0) {
    RNFail("Unable to set instance channel -- it has not been created\n");
    return;
  }

  // Set instance channel
  SetChannel(RGBD_INSTANCE_CHANNEL, image);
}



void RGBDImage::
SetCameraToWorld(const R3Affine& transformation)
{
  // Set camera_to_world transformation
  this->camera_to_world = transformation;

  // Invalidate bounding boxes
  InvalidateWorldBBox();
}



void RGBDImage::
SetExtrinsics(const R4Matrix& matrix)
{
  // Set extrinsics matrix
  this->camera_to_world.Reset(matrix.Inverse());

  // Invalidate bounding box
  InvalidateWorldBBox();
}



void RGBDImage::
SetIntrinsics(const R3Matrix& matrix)
{
  // Set intrinsics matrix
  this->intrinsics = matrix;

  // Invalidate bounding box
  InvalidateWorldBBox();
}



void RGBDImage::
SetTimestamp(RNScalar timestamp)
{
  // Set timestamp
  this->timestamp = timestamp;
}



void RGBDImage::
Transform(const R3Transformation& transformation)
{
  // Update camera_to_world transformation
  R3Affine t = R3identity_affine;
  t.Transform(transformation);
  t.Transform(camera_to_world);
  camera_to_world = t;

  // Invalidate bounding box
  InvalidateWorldBBox();
}



////////////////////////////////////////////////////////////////////////
// Transformation functions
////////////////////////////////////////////////////////////////////////

int RGBDImage::
TransformImageToCamera(const R2Point& image_position, R3Point& camera_position) const
{
  // Check image position
  int ix = (int) image_position[0];
  if ((ix < 0) || (ix >= NPixels(RN_X))) return 0; 
  int iy = (int) image_position[1];
  if ((iy < 0) || (iy >= NPixels(RN_Y))) return 0; 
  
  // Get/check depth (point sample to avoid interpolation)
  RNScalar depth = PixelDepth(ix, iy);
  if ((RNIsZero(depth)) || (depth == R2_GRID_UNKNOWN_VALUE)) return 0;

  // Get/check intrinsics matrix
  const R3Matrix& intrinsics_matrix = Intrinsics();
  if (RNIsZero(intrinsics_matrix[0][0])) return 0;
  if (RNIsZero(intrinsics_matrix[1][1])) return 0;
  
  // Transform from position in image coordinates to camera coordinates (where camera is looking down -Z, up is +Y, right is +X)
  camera_position[0] = (image_position[0] - intrinsics_matrix[0][2]) * depth / intrinsics_matrix[0][0];
  camera_position[1] = (image_position[1] - intrinsics_matrix[1][2]) * depth / intrinsics_matrix[1][1];
  camera_position[2] = -depth;

  // Return success
  return 1;
}



int RGBDImage::
TransformCameraToImage(const R3Point& camera_position, R2Point& image_position) const
{
  // Get/check depth
  RNScalar depth = -camera_position[2];
  if (RNIsNegativeOrZero(depth)) return 0;

  // Get/check intrinsics matrix
  const R3Matrix& intrinsics_matrix = Intrinsics();
  if (RNIsZero(intrinsics_matrix[0][0])) return 0;
  if (RNIsZero(intrinsics_matrix[1][1])) return 0;
  
  // Transform from position in image coordinates to camera coordinates (where camera is looking down -Z, up is +Y, right is +X)
  image_position[0] = intrinsics_matrix[0][2] + camera_position[0] * intrinsics_matrix[0][0] / depth;
  image_position[1] = intrinsics_matrix[1][2] + camera_position[1] * intrinsics_matrix[1][1] / depth;

  // Check image position
  if ((image_position[0] < 0) || (image_position[0] >= NPixels(RN_X))) return 0; 
  if ((image_position[1] < 0) || (image_position[1] >= NPixels(RN_Y))) return 0;

  // Return success
  return 1;
}



int RGBDImage::
TransformCameraToWorld(const R3Point& camera_position, R3Point& world_position) const
{
  // Transform from camera coordinates to world coordinates
  world_position = camera_position;
  world_position.Transform(CameraToWorld());
  return 1;
}



int RGBDImage::
TransformWorldToCamera(const R3Point& world_position, R3Point& camera_position) const
{
  // Transform from position in world coordinates to camera coordinates
  camera_position = world_position;
  camera_position.InverseTransform(CameraToWorld());
  if (RNIsPositiveOrZero(camera_position[2])) return 0;
  return 1;
}



int RGBDImage::
TransformWorldToImage(const R3Point& world_position, R2Point& image_position) const
{
  // Transform from position in world coordinates to image coordinates
  R3Point camera_position;
  if (!TransformWorldToCamera(world_position, camera_position)) return 0;
  if (!TransformCameraToImage(camera_position, image_position)) return 0;
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Draw functions
////////////////////////////////////////////////////////////////////////

void RGBDImage::
Draw(int color_scheme) const
{
  // Draw
  DrawCamera(color_scheme);
}



void RGBDImage::
DrawCamera(int color_scheme, RNLength radius) const
{
  // Get useful variables
  R3Point viewpoint = WorldViewpoint();
  R3Vector towards = WorldTowards();
  R3Vector up = WorldUp();

  // Load color
  if (color_scheme == RGBD_NO_COLOR_SCHEME) LoadColor(color_scheme);
  else if (color_scheme == RGBD_HIGHLIGHT_COLOR_SCHEME) LoadColor(color_scheme);
  else if (color_scheme != RGBD_NO_COLOR_SCHEME) LoadColor(RGBD_INDEX_COLOR_SCHEME);

  // Draw camera
  RNGrfxBegin(RN_GRFX_LINES);
  R3LoadPoint(viewpoint);
  R3LoadPoint(viewpoint + radius * towards);
  R3LoadPoint(viewpoint);
  R3LoadPoint(viewpoint + 0.5*radius * up);
  RNGrfxEnd();
}



void RGBDImage::
DrawBBox(int color_scheme) const
{
  // Load color
  if (color_scheme == RGBD_NO_COLOR_SCHEME) LoadColor(color_scheme);
  else if (color_scheme == RGBD_HIGHLIGHT_COLOR_SCHEME) LoadColor(color_scheme);
  else if (color_scheme != RGBD_NO_COLOR_SCHEME) LoadColor(RGBD_INDEX_COLOR_SCHEME);

  // Outline bounding box
  WorldBBox().Outline();
}



void RGBDImage::
DrawImage(int color_scheme, RNLength depth) const
{
  // Check image
  if ((width == 0) || (height == 0)) return;

  // Update/select opengl texture
  if (opengl_texture_id <= 0) ((RGBDImage *) this)->UpdateOpenGL();
  if (opengl_texture_id <= 0) return;
  glBindTexture(GL_TEXTURE_2D, opengl_texture_id);
  glEnable(GL_TEXTURE_2D);

  // Set color
  glDisable(GL_LIGHTING);
  RNLoadRgb(1.0, 1.0, 1.0);

  // Check depth
  if (depth > 0) {
    // Draw textured polygon on view plane
    R3Point viewpoint = WorldViewpoint();
    R3Vector towards = WorldTowards();
    R3Vector right = WorldRight();
    R3Vector up = WorldUp();
    R3Point c = viewpoint + depth * towards;
    R3Vector dx = (depth * 0.5*width / intrinsics[0][0]) * right;
    R3Vector dy = (depth * 0.5*height / intrinsics[1][1]) * up;
    c -= dx*(intrinsics[0][2] - 0.5*width)/width;
    c -= dy*(intrinsics[1][2] - 0.5*height)/height;
    RNGrfxBegin(RN_GRFX_QUADS);
    R3LoadTextureCoords(0, 0);
    R3LoadPoint(c - dx - dy);
    R3LoadTextureCoords(1, 0);
    R3LoadPoint(c + dx - dy);
    R3LoadTextureCoords(1, 1);
    R3LoadPoint(c + dx + dy);
    R3LoadTextureCoords(0, 1);
    R3LoadPoint(c - dx + dy);
    RNGrfxEnd();
  }
  else {
    // Draw textured polygon on screen plane (in pixels)
    RNGrfxBegin(RN_GRFX_QUADS);
    R3LoadTextureCoords(0, 0);
    R3LoadPoint(0, 0, 0);
    R3LoadTextureCoords(1, 0);
    R3LoadPoint(width-1, 0, 0);
    R3LoadTextureCoords(1, 1);
    R3LoadPoint(width-1, height-1, 0);
    R3LoadTextureCoords(0, 1);
    R3LoadPoint(0, height-1, 0);
    RNGrfxEnd();
  }
  
  // Unselect opengl texture
  glDisable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
}



void RGBDImage::
DrawPoints(int color_scheme, int skip) const
{
  // Push transformation
  CameraToWorld().Push();

  // Enable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
  }

  // Draw color
  LoadColor(color_scheme);

  // Set point size
  glPointSize(2);

  // Draw points
  if (color_scheme != RGBD_RENDER_COLOR_SCHEME) {
    RNGrfxBegin(RN_GRFX_POINTS);
    R3Point world_position;
    for (int ix = 0; ix < NPixels(RN_X); ix += skip) {
      for (int iy = 0; iy < NPixels(RN_Y); iy += skip) {
        if (RGBDTransformImageToCamera(R2Point(ix+0.5, iy+0.5), world_position, this)) {
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix, iy));
          R3LoadPoint(world_position);
        }
      }
    }
    RNGrfxEnd();
  }
  else {
    RNGrfxBegin(RN_GRFX_POINTS);
    R3Point world_position, p1, p2;
    for (int ix = 0; ix < NPixels(RN_X); ix += skip) {
      for (int iy = 0; iy < NPixels(RN_Y); iy += skip) {
        if (!RGBDTransformImageToCamera(R2Point(ix+0.5, iy+0.5), world_position, this)) continue;
        if (!RGBDTransformImageToCamera(R2Point(ix+1, iy), p1, this)) continue;
        if (!RGBDTransformImageToCamera(R2Point(ix, iy+1), p2, this)) continue;
        R3Plane plane(world_position, p1, p2);
        R3LoadNormal(plane.Normal());
        R3LoadPoint(world_position);
      }
    }
    RNGrfxEnd();
  }

  // Reset point size
  glPointSize(1);
  
  // Disable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
  }

  // Pop transformation
  CameraToWorld().Pop();
}



void RGBDImage::
DrawSurfels(int color_scheme, int skip) const
{
  // Get info about current view
  GLint viewport[4];
  GLdouble modelview_matrix[16];
  GLdouble projection_matrix[16];
  R3Point c0(0, 0, 0), w0;
  R3Point c1(0, 0, -1), w1;
  R3Point c2(1, 0, -1), w2;
  R3Point c3(0, 1, -1), w3;
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
  glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);
  gluUnProject(c0[0], c0[1], c0[2], modelview_matrix, projection_matrix, viewport, &(w0[0]), &(w0[1]), &(w0[2]));
  gluUnProject(c1[0], c1[1], c1[2], modelview_matrix, projection_matrix, viewport, &(w1[0]), &(w1[1]), &(w1[2]));
  gluUnProject(c2[0], c2[1], c2[2], modelview_matrix, projection_matrix, viewport, &(w2[0]), &(w2[1]), &(w2[2]));
  gluUnProject(c3[0], c3[1], c3[2], modelview_matrix, projection_matrix, viewport, &(w3[0]), &(w3[1]), &(w3[2]));
  R3Vector view_towards = w1 - w0; view_towards.Normalize();
  R3Vector view_right = w2 - w1; view_right.Normalize();
  R3Vector view_up = w3 - w1; view_up.Normalize();

  // Enable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
  }

  // Draw color
  LoadColor(color_scheme);

  // Draw little quads
  RNGrfxBegin(RN_GRFX_QUADS);
  R3Point world_position;
  for (int ix = 0; ix < NPixels(RN_X)-1; ix += skip) {
    for (int iy = 0; iy < NPixels(RN_Y)-1; iy += skip) {
      RNScalar depth = PixelDepth(ix, iy);
      if (RNIsZero(depth)) continue;
      if (!RGBDTransformImageToWorld(R2Point(ix, iy), world_position, this)) continue;
      RNScalar r = depth / Intrinsics()[0][0];
      if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix, iy));
      R3LoadPoint(world_position - r*view_right - r*view_up);
      R3LoadPoint(world_position + r*view_right - r*view_up);
      R3LoadPoint(world_position + r*view_right + r*view_up);
      R3LoadPoint(world_position - r*view_right + r*view_up);
    }
  }
  RNGrfxEnd();
  
  // Disable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
  }
}



void RGBDImage::
DrawQuads(int color_scheme, int skip) const
{
  // Get info about current view
  GLint viewport[4];
  GLdouble modelview_matrix[16];
  GLdouble projection_matrix[16];
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview_matrix);
  glGetDoublev(GL_PROJECTION_MATRIX, projection_matrix);

  // Push transformation
  CameraToWorld().Push();

  // Enable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
  }

  // Draw color
  LoadColor(color_scheme);

  // Draw quads
  RNGrfxBegin(RN_GRFX_TRIANGLES);
  for (int ix = 0; ix < NPixels(RN_X)-skip; ix += skip) {
    for (int iy = 0; iy < NPixels(RN_Y)-skip; iy += skip) {
      R3Point p1, p2, p3, p4;
      if (!RGBDTransformImageToCamera(R2Point(ix, iy), p1, this)) continue;
      if (!RGBDTransformImageToCamera(R2Point(ix+skip, iy), p2, this)) continue;
      if (!RGBDTransformImageToCamera(R2Point(ix+skip, iy+skip), p3, this)) continue;
      if (!RGBDTransformImageToCamera(R2Point(ix, iy+skip), p4, this)) continue;
      R3Plane plane(p1, p2, p3);
      RNScalar dot = plane.Normal().Dot(p1.Vector());
      if (dot < -0.5) {
        if ((fabs(p1.Z() - p2.Z())/-p1.Z() < 0.05) &&
            (fabs(p2.Z() - p3.Z())/-p2.Z() < 0.05) &&
            (fabs(p3.Z() - p1.Z())/-p3.Z() < 0.05)) {
          R3LoadNormal(plane.Normal());
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix, iy));
          R3LoadPoint(p1);
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix+skip, iy));
          R3LoadPoint(p2);
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix+skip, iy+skip));
          R3LoadPoint(p3);
        }
        if ((fabs(p1.Z() - p3.Z())/-p1.Z() < 0.05) &&
            (fabs(p3.Z() - p4.Z())/-p3.Z() < 0.05) &&
            (fabs(p4.Z() - p1.Z())/-p4.Z() < 0.05)) {
          R3LoadNormal(plane.Normal());
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix, iy));
          R3LoadPoint(p1);
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix+skip, iy+skip));
          R3LoadPoint(p3);
          if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) RNLoadRgb(PixelColor(ix, iy+skip));
          R3LoadPoint(p4);
        }
      }
    }
  }
  RNGrfxEnd();

  // Disable lighting and material
  if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
  }

  // Pop transformation
  CameraToWorld().Pop();
}



void RGBDImage::
LoadColor(int color_scheme) const
{
  // Check color scheme
  if (color_scheme == RGBD_INDEX_COLOR_SCHEME) {
    // Load color encoding image index
    int k = 65535.0 * (ConfigurationIndex()+1) / configuration->NImages();
    unsigned char r = 0;
    unsigned char g = (k >> 8) & 0xFF;
    unsigned char b = k & 0xFF;
    RNLoadRgb(r, g, b);
  }
  else if (color_scheme == RGBD_RENDER_COLOR_SCHEME) {
    // Load white
    RNLoadRgb(1.0, 1.0, 1.0);
  }
  else if (color_scheme == RGBD_PHOTO_COLOR_SCHEME) {
    // Load white
    RNLoadRgb(1.0, 1.0, 1.0);
  }
  else if (color_scheme == RGBD_HIGHLIGHT_COLOR_SCHEME) {
    // Load highlight color
    RNLoadRgb(1.0, 1.0, 0.0);
  }
}



////////////////////////////////////////////////////////////////////////
// Image processing functions
////////////////////////////////////////////////////////////////////////

#if 0
static int
MaskBoundaries(R2Grid& depth_image, RNScalar depth_threshold = 0.1, RNScalar pixel_erosion_distance = 100)
{
  // Create copy of depth image with holes filled
  R2Grid filled_depth_image(depth_image);
  filled_depth_image.Substitute(0, R2_GRID_UNKNOWN_VALUE);
  filled_depth_image.FillHoles();

  // Mark interior boundaries, silhouettes, and shadows
  for (int i = 1; i < depth_image.XResolution()-1; i++) {
    for (int j = 1; j < depth_image.YResolution()-1; j++) {
      // Get original depth
      RNScalar depth = depth_image.GridValue(i, j);
      if (RNIsNegativeOrZero(depth)) continue;

      // Get filled depth
      depth = filled_depth_image.GridValue(i, j);
      if (RNIsNegativeOrZero(depth)) continue;

      // Check depth relative to horizontal neighbors
      for (int k = 0; k < 4; k++) {
        int s = (k < 3) ? -1 : 0;
        int t = (k < 3) ? k-1 : -1;

        // Get depth on one side
        RNScalar depthA = filled_depth_image.GridValue(i-s, j-t);
        if (RNIsNegativeOrZero(depthA)) continue;

        // Get depth on other side
        RNScalar depthB = filled_depth_image.GridValue(i+s, j+t);
        if (RNIsNegativeOrZero(depthB)) continue;

        // Check differences of depth for shadow/silhouette
        RNScalar deltaA = depth - depthA;
        RNScalar deltaB = depthB - depth;
        RNScalar threshold = depth * depth_threshold;
        if (threshold < 0.1) threshold = 0.1;
        if (deltaA < -threshold) {
          if (deltaA < 4*deltaB) {
            depth_image.SetGridValue(i-s, j-t, 0.0);
            depth_image.SetGridValue(i, j, 0.0);
          }
        }
        else if (deltaA > threshold) {
          if (deltaA > 4*deltaB) {
            depth_image.SetGridValue(i-s, j-t, 0.0);
            depth_image.SetGridValue(i, j, 0.0);
          }
        }
        if (deltaB < -threshold) {
          if (deltaB < 4*deltaA) {
            depth_image.SetGridValue(i+s, j+t, 0.0);
            depth_image.SetGridValue(i, j, 0.0);
          }
        }
        else if (deltaB > threshold) {
          if (deltaB > 4*deltaA) {
            depth_image.SetGridValue(i+s, j+t, 0.0);
            depth_image.SetGridValue(i, j, 0.0);
          }
        }
      }
    }
  }

  // Erode by 
  R2Grid mask(depth_image);
  mask.Erode(pixel_erosion_distance);
  depth_image.Mask(mask);

  // Return success 
  return 1;
}
#endif



#if 0
static int
FillMissingDepthValues(R2Grid& depth_image,
  int pixel_radius = 4, int min_pixel_count = 6,
  RNLength max_delta_d = 0.2, RNLength max_closest_d = 0.025)
{
  // Check parameters
  if ((max_delta_d == 0) && (max_closest_d == 0)) return 1;

  // Fill missing depth values
  R2Grid copy(grid);
  for (int ix = 0; ix < grid.XResolution(); ix++) {
    for (int iy = 0; iy < grid.YResolution(); iy++) {
      RNScalar depth = copy.GridValue(ix, iy);
      if (RNIsPositive(depth)) continue;

      // Gather statistics from neighborhood
      int count = 0;
      RNScalar sum = 0;
      int closest_dd = INT_MAX;
      RNScalar closest_d = 0;
      RNScalar min_d = FLT_MAX;
      RNScalar max_d = -FLT_MAX;
      for (int s = -pixel_radius; s <= pixel_radius; s++) {
        if ((ix + s < 0) || (ix + s >= grid.XResolution())) continue;
        for (int t = -pixel_radius; t <= pixel_radius; t++) {
          if ((iy + t < 0) || (iy + t >= grid.YResolution())) continue;
          RNScalar d = copy.GridValue(ix+s, iy+t);
          if (RNIsNegativeOrZero(d)) continue;
          int dd = s*s + t*t;
          if (dd < closest_dd) { closest_dd = dd; closest_d = d; }
          if (d < min_d) min_d = d;
          if (d > max_d) max_d = d;
          sum += d;
          count++;
        }
      }

      // Fill in missing depth value with average if on planar surface
      if (count >= min_pixel_count) {
        if ((max_d - min_d) < max_delta_d) {
          RNScalar mean = sum / count;
          if (RNIsEqual(closest_d, mean, max_closest_d)) {
            grid.SetGridValue(ix, iy, mean);
          }
        }
      }
    }
  }

  // Return success 
  return 1;
}
#endif



////////////////////////////////////////////////////////////////////////
// Access convenience functions
////////////////////////////////////////////////////////////////////////

R2Image RGBDImage::
ColorChannels(void) const
{
  // Initialize image
  R2Image image(NPixels(RN_X), NPixels(RN_Y), 3);

  // Fill image
  R2Grid *red_channel = Channel(RGBD_RED_CHANNEL);
  R2Grid *green_channel = Channel(RGBD_GREEN_CHANNEL);
  R2Grid *blue_channel = Channel(RGBD_BLUE_CHANNEL);
  for (int j = 0; j < image.Height(); j++) {
    for (int i = 0; i < image.Width(); i++) {
      RNScalar r = red_channel->GridValue(i, j);
      RNScalar g = green_channel->GridValue(i, j);
      RNScalar b = blue_channel->GridValue(i, j);
      image.SetPixelRGB(i, j, RNRgb(r, g, b));
    }
  }

  // Return image
  return image;
}



////////////////////////////////////////////////////////////////////////
// Create/read/write/release functions
////////////////////////////////////////////////////////////////////////

int RGBDImage::
CreateColorChannels(const R2Image& image)
{
  // Create color channels
  while (channels.NEntries() <= RGBD_BLUE_CHANNEL) channels.Insert(NULL);
  if (!channels[RGBD_RED_CHANNEL]) channels[RGBD_RED_CHANNEL] = new R2Grid(image.Width(), image.Height());
  if (!channels[RGBD_GREEN_CHANNEL]) channels[RGBD_GREEN_CHANNEL] = new R2Grid(image.Width(), image.Height());
  if (!channels[RGBD_BLUE_CHANNEL]) channels[RGBD_BLUE_CHANNEL] = new R2Grid(image.Width(), image.Height());

  // Update resident count
  color_resident_count++;

  // Initialize color channels
  SetColorChannels(image);

  // Set width and height
  this->width = image.Width();
  this->height = image.Height();

  // Return success
  return 1;
}



int RGBDImage::
CreateDepthChannel(const R2Grid& image)
{
  // Create depth channel
  return CreateChannel(RGBD_DEPTH_CHANNEL, image);
}



int RGBDImage::
CreateCategoryChannel(const R2Grid& image)
{
  // Create category channel
  return CreateChannel(RGBD_CATEGORY_CHANNEL, image);
}



int RGBDImage::
CreateInstanceChannel(const R2Grid& image)
{
  // Create instance channel
  return CreateChannel(RGBD_INSTANCE_CHANNEL, image);
}



int RGBDImage::
CreateChannel(int channel_index, const R2Grid& image)
{
  // Create channel
  while (channels.NEntries() <= channel_index) channels.Insert(NULL);
  if (!channels[channel_index]) channels[channel_index] = new R2Grid(image);

  // Update resident count
  if (channel_index == RGBD_DEPTH_CHANNEL) depth_resident_count++;
  else if (channel_index == RGBD_BLUE_CHANNEL) color_resident_count++;

  // Initialize channel
  SetChannel(channel_index, image);

  // Set width and height
  this->width = image.XResolution();
  this->height = image.YResolution();

  // Return success
  return 1;
}



int RGBDImage::
ReadChannels(void)
{
  // Read all channels
  if (!ReadColorChannels()) return 0;
  if (!ReadDepthChannel()) return 0;
  if (!ReadCategoryChannel()) return 0;
  if (!ReadInstanceChannel()) return 0;
  return 1;
}



int RGBDImage::
ReadColorChannels(void)
{
  // Check if already resident
  if (color_resident_count > 0) {
    color_resident_count++;
    return 1;
  }

  // Initialize image
  R2Image color_image(width, height, 3);

  // Check filename
  if (color_filename) {
    // Get full filename
    char full_filename[4096];
    const char *dirname = (configuration) ? configuration->ColorDirectory() : NULL;
    if (dirname) sprintf(full_filename, "%s/%s", dirname, color_filename);
    else sprintf(full_filename, "%s", color_filename);

    // Read color image
    if (!color_image.Read(full_filename)) return 0;

    // Resize color image if necessary
    if ((width > 0) && (height > 0) &&
        ((width != color_image.Width()) || (height != color_image.Height()))) {
      R3Matrix tmp(R3identity_matrix);
      RGBDResampleColorImage(color_image, tmp, width, height);
    }
  }

  // Create color channels
  return CreateColorChannels(color_image);
}



int RGBDImage::
ReadDepthChannel(void)
{
  // Check if already resident/update read count
  if (depth_resident_count > 0) {
    depth_resident_count++;
    return 1;
  }

  // Initialize image
  R2Grid depth_image(width, height);

  // Check filename
  if (depth_filename) {
    // Get full filename
    char full_filename[4096];
    const char *dirname = (configuration) ? configuration->DepthDirectory() : NULL;
    if (dirname) sprintf(full_filename, "%s/%s", dirname, depth_filename);
    else sprintf(full_filename, "%s", depth_filename);

    // Read depth image
    if (!depth_image.ReadFile(full_filename)) return 0;

    // Shift 3 bits (to compensate for shift done by SUNRGBD capture)
    if (configuration && configuration->DatasetFormat()) {
      if (!strcmp(configuration->DatasetFormat(), "sun3d") || !strcmp(configuration->DatasetFormat(), "sunrgbd")) {
        for (int i = 0; i < depth_image.NEntries(); i++) {
          unsigned int d = (unsigned int) (depth_image.GridValue(i) + 0.5);
          d = ((d >> 3) & 0x1FFF) | ((d & 0x7) << 13);
          depth_image.SetGridValue(i, d);
        }
      }
    }

    // Substite unknown for zero
    depth_image.Substitute(0, R2_GRID_UNKNOWN_VALUE);
    // MaskBoundaries(depth_image);

    // Perform dataset-dependent processing
    if (strstr(depth_filename, ".png")) {
      depth_image.Multiply(0.001);
      if (configuration && configuration->DatasetFormat()) {
        if (!strcmp(configuration->DatasetFormat(), "matterport")) {
          depth_image.Multiply(0.25);
        }
        else if (!strcmp(configuration->DatasetFormat(), "tum")) {
          depth_image.Multiply(0.2);
        }
        else if (!strcmp(configuration->DatasetFormat(), "gsv")) {
          depth_image.Multiply(0.5);
          depth_image.Pow(2);
        }
        else if (!strcmp(configuration->DatasetFormat(), "icl")) {
          depth_image.Multiply(0.2);
        }
        else if (!strcmp(configuration->DatasetFormat(), "sumo")) {
          if (RNIsNotZero(intrinsics[0][0]) && RNIsNotZero(intrinsics[1][1])) {
            for (int ix = 0; ix < depth_image.XResolution(); ix++) {
              for (int iy = 0; iy < depth_image.YResolution(); iy++) {
                RNScalar value = depth_image.GridValue(ix, iy);
                if (value == 0) continue;
                RNScalar distance = 65.535 * 0.3 / value;
                RNScalar dx = (ix - intrinsics[0][2]) / intrinsics[0][0];
                RNScalar dy = (iy - intrinsics[1][2]) / intrinsics[1][1];
                RNScalar r = sqrt(dx*dx + dy*dy);
                RNScalar depth =  distance / sqrt(1 + r*r);
                depth_image.SetGridValue(ix, iy, depth);
              }
            }
          }
        }
      }
    }

#if 0
    // Smooth depth image
    if (!configuration || !configuration->DatasetFormat() ||
        (strcmp(configuration->DatasetFormat(), "processed") &&
         strcmp(configuration->DatasetFormat(), "matterport") &&         
         strcmp(configuration->DatasetFormat(), "scannet"))) {
      RNScalar d_sigma_fraction = 0.015;
      RNScalar xy_sigma = 3 * depth_image.XResolution() / 640.0;
      depth_image.BilateralFilter(xy_sigma, d_sigma_fraction, TRUE);
    }
#endif
    
    // Resize depth image if necessary
    if ((width > 0) && (height > 0) && ((width != depth_image.XResolution()) || (height != depth_image.YResolution()))) {
      R3Matrix tmp(R3identity_matrix);
      RGBDResampleDepthImage(depth_image, tmp, width, height);
    }
  }
  
  // Create depth channel
  return CreateDepthChannel(depth_image);
}



int RGBDImage::
ReadCategoryChannel(void)
{
  // Check if already resident/update read count
  if (category_resident_count > 0) {
    category_resident_count++;
    return 1;
  }

  // Initialize image
  R2Grid category_image(width, height);

  // Check filename
  if (category_filename) {
    // Get full filename
    char full_filename[4096];
    const char *dirname = (configuration) ? configuration->CategoryDirectory() : NULL;
    if (dirname) sprintf(full_filename, "%s/%s", dirname, category_filename);
    else sprintf(full_filename, "%s", category_filename);

    // Read category image
    if (!category_image.ReadFile(full_filename)) return 0;
  }
  
  // Create category channel
  return CreateCategoryChannel(category_image);
}



int RGBDImage::
ReadInstanceChannel(void)
{
  // Check if already resident/update read count
  if (instance_resident_count > 0) {
    instance_resident_count++;
    return 1;
  }

  // Initialize image
  R2Grid instance_image(width, height);

  // Check filename
  if (instance_filename) {
    // Get full filename
    char full_filename[4096];
    const char *dirname = (configuration) ? configuration->InstanceDirectory() : NULL;
    if (dirname) sprintf(full_filename, "%s/%s", dirname, instance_filename);
    else sprintf(full_filename, "%s", instance_filename);

    // Read instance image
    if (!instance_image.ReadFile(full_filename)) return 0;
  }
  
  // Create instance channel
  return CreateInstanceChannel(instance_image);
}



int RGBDImage::
WriteChannels(void)
{
  // Write all channels
  if (!WriteColorChannels()) return 0;
  if (!WriteDepthChannel()) return 0;
  if (!WriteCategoryChannel()) return 0;
  if (!WriteInstanceChannel()) return 0;
  return 1;
}



int RGBDImage::
WriteColorChannels(void)
{
  // Check filename
  if (!color_filename) return 0;
  if (NChannels() <= RGBD_BLUE_CHANNEL) return 0;

  // Get full filename
  char full_filename[4096];
  const char *dirname = (configuration) ? configuration->ColorDirectory() : NULL;
  if (dirname) sprintf(full_filename, "%s/%s", dirname, color_filename);
  else sprintf(full_filename, "%s", color_filename);

  // Compute color image
  R2Image rgb_image(width, height, 3);
  for (int iy = 0; iy < height; iy++) {
    for (int ix = 0; ix < width; ix++) {
      RNRgb rgb = PixelColor(ix, iy);
      rgb_image.SetPixelRGB(ix, iy, rgb);
    }
  }

  // Write color image
  if (!rgb_image.Write(color_filename)) return 0;

  // Return success
  return 1;
}



int RGBDImage::
WriteDepthChannel(void)
{
  // Check filename
  if (!depth_filename) return 0;
  if (NChannels() <= RGBD_DEPTH_CHANNEL) return 0;

  // Get full filename
  char full_filename[4096];
  const char *dirname = (configuration) ? configuration->DepthDirectory() : NULL;
  if (dirname) sprintf(full_filename, "%s/%s", dirname, depth_filename);
  else sprintf(full_filename, "%s", depth_filename);
  
  // Get depth image  
  R2Grid depth_image(*(channels[RGBD_DEPTH_CHANNEL]));
  if (strstr(depth_filename, ".png")) {
    depth_image.Multiply(1000.0);
    if (configuration && configuration->DatasetFormat()) {
      if (!strcmp(configuration->DatasetFormat(), "matterport")) {
        depth_image.Multiply(4);
      }
      else if (!strcmp(configuration->DatasetFormat(), "tum")) {
        depth_image.Multiply(5);
      }
    }
  }
    
  // Write depth image  
  if (!depth_image.WriteFile(depth_filename)) return 0;

  // Return success
  return 1;
}



int RGBDImage::
WriteCategoryChannel(void)
{
  // Check filename
  if (!category_filename) return 0;
  if (NChannels() <= RGBD_CATEGORY_CHANNEL) return 0;

  // Get full filename
  char full_filename[4096];
  const char *dirname = (configuration) ? configuration->CategoryDirectory() : NULL;
  if (dirname) sprintf(full_filename, "%s/%s", dirname, category_filename);
  else sprintf(full_filename, "%s", category_filename);
  
  // Write category image  
  if (!channels[RGBD_CATEGORY_CHANNEL]->WriteFile(category_filename)) return 0;

  // Return success
  return 1;
}



int RGBDImage::
WriteInstanceChannel(void)
{
  // Check filename
  if (!instance_filename) return 0;
  if (NChannels() <= RGBD_INSTANCE_CHANNEL) return 0;

  // Get full filename
  char full_filename[4096];
  const char *dirname = (configuration) ? configuration->InstanceDirectory() : NULL;
  if (dirname) sprintf(full_filename, "%s/%s", dirname, instance_filename);
  else sprintf(full_filename, "%s", instance_filename);
  
  // Write instance image  
  if (!channels[RGBD_INSTANCE_CHANNEL]->WriteFile(instance_filename)) return 0;

  // Return success
  return 1;
}



int RGBDImage::
ReleaseChannels(void)
{
  // Release all channels
  if (!ReleaseColorChannels()) return 0;
  if (!ReleaseDepthChannel()) return 0;
  if (!ReleaseCategoryChannel()) return 0;
  if (!ReleaseInstanceChannel()) return 0;
  return 1;
}



int RGBDImage::
ReleaseColorChannels(void)
{
  // Check/update read count
  if (--color_resident_count > 0) return 1;

  // Write color channel before releasing it ???
  // if (!WriteColorChannels()) return 0;

  // Delete color channels
  for (int channel_index = RGBD_RED_CHANNEL; channel_index <= RGBD_BLUE_CHANNEL; channel_index++) {
    if (NChannels() <= channel_index) break;
    if (!channels[channel_index]) continue;
    delete channels[channel_index];
    channels[channel_index] = NULL;
  }

  // Return success;
  return 1;
}



int RGBDImage::
ReleaseDepthChannel(void)
{
  // Check/update read count
  if (--depth_resident_count > 0) return 1;

  // Write depth channel before releasing it ???
  // if (!WriteDepthChannels()) return 0;

  // Delete depth channel
  if (NChannels() <= RGBD_DEPTH_CHANNEL) return 0;
  if (!channels[RGBD_DEPTH_CHANNEL]) return 0;
  delete channels[RGBD_DEPTH_CHANNEL];
  channels[RGBD_DEPTH_CHANNEL] = NULL;

  // Return success;
  return 1;
}



int RGBDImage::
ReleaseCategoryChannel(void)
{
  // Check/update read count
  if (--category_resident_count > 0) return 1;

  // Write category channel before releasing it ???
  // if (!WriteCategoryChannels()) return 0;

  // Delete category channel
  if (NChannels() <= RGBD_CATEGORY_CHANNEL) return 0;
  if (!channels[RGBD_CATEGORY_CHANNEL]) return 0;
  delete channels[RGBD_CATEGORY_CHANNEL];
  channels[RGBD_CATEGORY_CHANNEL] = NULL;

  // Return success;
  return 1;
}



int RGBDImage::
ReleaseInstanceChannel(void)
{
  // Check/update read count
  if (--instance_resident_count > 0) return 1;

  // Write instance channel before releasing it ???
  // if (!WriteInstanceChannels()) return 0;

  // Delete instance channel
  if (NChannels() <= RGBD_INSTANCE_CHANNEL) return 0;
  if (!channels[RGBD_INSTANCE_CHANNEL]) return 0;
  delete channels[RGBD_INSTANCE_CHANNEL];
  channels[RGBD_INSTANCE_CHANNEL] = NULL;

  // Return success;
  return 1;
}



void RGBDImage::
SetName(const char *name)
{
  // Set filename
  if (this->name) free(this->name);
  if (name && strcmp(name, "-")) this->name = RNStrdup(name);
  else this->name = NULL;
}



void RGBDImage::
SetColorFilename(const char *filename)
{
  // Set filename
  if (color_filename) free(color_filename);
  if (filename && strcmp(filename, "-")) color_filename = RNStrdup(filename);
  else color_filename = NULL;

  // Set name
  if (!name && filename) {
    char buffer[1024];
    strncpy(buffer, filename, 1023);
    char *startp = strrchr(buffer, '/');
    if (startp) startp++;
    else startp = buffer;
    char *endp = strrchr(startp, '.');
    if (endp) *endp = '\0';
    name = RNStrdup(startp);
  }
}



void RGBDImage::
SetDepthFilename(const char *filename)
{
  // Set filename
  if (depth_filename) free(depth_filename);
  if (filename && strcmp(filename, "-")) depth_filename = RNStrdup(filename);
  else depth_filename = NULL;

  // Set name
  if (!name && filename && strcmp(filename, "-")) {
    char buffer[1024];
    strncpy(buffer, filename, 1023);
    char *startp = strrchr(buffer, '/');
    if (startp) startp++;
    else startp = buffer;
    char *endp = strrchr(startp, '.');
    if (endp) *endp = '\0';
    name = RNStrdup(startp);
  }
}



void RGBDImage::
SetCategoryFilename(const char *filename)
{
  // Set filename
  if (category_filename) free(category_filename);
  if (filename && strcmp(filename, "-")) category_filename = RNStrdup(filename);
  else category_filename = NULL;

  // Set name
  if (!name && filename && strcmp(filename, "-")) {
    char buffer[1024];
    strncpy(buffer, filename, 1023);
    char *startp = strrchr(buffer, '/');
    if (!startp) startp = buffer;
    char *endp = strrchr(startp, '.');
    if (endp) *endp = '\0';
    name = RNStrdup(startp);
  }
}



void RGBDImage::
SetInstanceFilename(const char *filename)
{
  // Set filename
  if (instance_filename) free(instance_filename);
  if (filename && strcmp(filename, "-")) instance_filename = RNStrdup(filename);
  else instance_filename = NULL;

  // Set name
  if (!name && filename) {
    char buffer[1024];
    strncpy(buffer, filename, 1023);
    char *startp = strrchr(buffer, '/');
    if (!startp) startp = buffer;
    char *endp = strrchr(startp, '.');
    if (endp) *endp = '\0';
    name = RNStrdup(startp);
  }
}



////////////////////////////////////////////////////////////////////////
// Update functions
////////////////////////////////////////////////////////////////////////

void RGBDImage::
InvalidateWorldBBox(void)
{
  // Mark bounding box for recomputation
  world_bbox.Reset(R3Point(FLT_MAX, FLT_MAX, FLT_MAX), R3Point(-FLT_MAX, -FLT_MAX, -FLT_MAX));

  // Invalidate the configuration's bounding box
  if (configuration) configuration->InvalidateWorldBBox();
}



void RGBDImage::
InvalidateOpenGL(void) 
{
  // Delete opengl texture
  if (opengl_texture_id > 0) {
    GLuint i = opengl_texture_id;
    glDeleteTextures(1, &i);
  }

  // Reset opengl identifier
  opengl_texture_id = -1;
}


  
void RGBDImage::
UpdateOpenGL(void) 
{
  // Check identifier
  if (opengl_texture_id > 0) return;
  
  // Create identifier
  GLuint identifier;
  glGenTextures(1, &identifier);

  // Read color channels
  // ReadColorChannels();
  
  // Create temporary R2Image
  R2Image rgb_image(width, height, 3);
  for (int iy = 0; iy < height; iy++) {
    for (int ix = 0; ix < width; ix++) {
      rgb_image.SetPixelRGB(ix, iy, PixelColor(ix, iy));
    }
  }

  // Release color channels
  // ReleaseColorChannels();

  // Define texture
  glBindTexture(GL_TEXTURE_2D, identifier);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  gluBuild2DMipmaps(GL_TEXTURE_2D, 3, rgb_image.Width(), rgb_image.Height(),
    GL_RGB, GL_UNSIGNED_BYTE, (const unsigned char *) rgb_image.Pixels());

  // Remember identifier
  opengl_texture_id = identifier;
}



////////////////////////////////////////////////////////////////////////
// Mesh creation function
////////////////////////////////////////////////////////////////////////

int RGBDImage::
ComputeMesh(R3Mesh& mesh, RNLength max_depth, RNLength max_silhouette_factor)
{
  // Read the channels
  ReadChannels();

  // Allocate temporary array of vertex pointers
  int nvertices = NPixels(RN_X) * NPixels(RN_Y);
  R3MeshVertex **vertices = new R3MeshVertex * [ nvertices ];
  for (int i = 0; i < nvertices; i++) vertices[i] = NULL;

  // Compute boundary image ???
  R2Grid boundary_image(NPixels(RN_X), NPixels(RN_Y));
  if (!RGBDCreateBoundaryChannel(this, boundary_image, max_silhouette_factor)) return 0;
  boundary_image.Substitute(R2_GRID_UNKNOWN_VALUE, 0);
         
  // Create vertices
  for (int j = 0; j < NPixels(RN_Y); j++) {
    for (int i = 0; i < NPixels(RN_X); i++) {
      // Check depth
      RNScalar depth = PixelDepth(i, j);
      if (depth == R2_GRID_UNKNOWN_VALUE) continue;
      if (RNIsNegativeOrZero(depth)) continue;
      if ((max_depth > 0) && (depth > max_depth)) continue;

      // Check if on border ???
      unsigned int b = (unsigned int) (boundary_image.GridValue(i, j) + 0.5);
      if (b & RGBD_BORDER_BOUNDARY) continue;
      // if (b) continue; 
      
      // Compute vertex info
      R3Point position = PixelWorldPosition(i, j);
      RNRgb color = PixelColor(i, j);
      R2Point texcoords(i, j);

      // Create vertex
      int vertex_index = j*NPixels(RN_X) + i;
      vertices[vertex_index] = mesh.CreateVertex(position, R3zero_vector, color, texcoords);
    }
  }

  // Create faces
  for (int i = 0; i < NPixels(RN_X)-1; i++) {
    for (int j = 0; j < NPixels(RN_Y)-1; j++) {
      // Get vertices
      R3MeshVertex *v00 = vertices[(j+0)*NPixels(RN_X) + (i+0)];
      R3MeshVertex *v01 = vertices[(j+1)*NPixels(RN_X) + (i+0)];
      R3MeshVertex *v10 = vertices[(j+0)*NPixels(RN_X) + (i+1)];
      R3MeshVertex *v11 = vertices[(j+1)*NPixels(RN_X) + (i+1)];

      // Get boundary flags
      unsigned int b00 = (unsigned int) (boundary_image.GridValue(i, j) + 0.5);
      unsigned int b10 = (unsigned int) (boundary_image.GridValue(i+1, j) + 0.5);
      unsigned int b01 = (unsigned int) (boundary_image.GridValue(i, j+1) + 0.5);
      unsigned int b11 = (unsigned int) (boundary_image.GridValue(i+1, j+1) + 0.5);
      unsigned int bad_combination = RGBD_SILHOUETTE_BOUNDARY | RGBD_SHADOW_BOUNDARY;

      // Create faces
      if (!v00) {
        if (v10 && v11 && v01) {
          if (((b10 | b11 | b01) & bad_combination) != bad_combination) {
            mesh.CreateFace(v10, v11, v01);
          }
        }
      }
      else if (!v10) {
        if (v00 && v11 && v01) {
          if (((b00 | b11 | b01) & bad_combination) != bad_combination) {
            mesh.CreateFace(v00, v11, v01);
          }
        }
      }
      else if (!v01) {
        if (v00 && v10 && v11) {
          if (((b00 | b10 | b11) & bad_combination) != bad_combination) {
            mesh.CreateFace(v00, v10, v11);
          }
        }
      }
      else if (!v11) {
        if (v00 && v10 && v01) {
          if (((b00 | b10 | b01) & bad_combination) != bad_combination) {
            mesh.CreateFace(v00, v10, v01);
          }
        }
      }
      else {
        assert(v00 && v01 && v10 && v11);
        if (((b00 | b10 | b11) & bad_combination) != bad_combination) {
          mesh.CreateFace(v00, v10, v11);
        }
        if (((b00 | b11 | b01) & bad_combination) != bad_combination) {
          mesh.CreateFace(v00, v11, v01);
        }
      } 
    }
  }

  // Delete temporary array of vertex pointers
  delete [] vertices;

  // Release the channels
  ReleaseChannels();

  // Return success
  return 1;
}



} // namespace gaps
