# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os
import bpy
import numpy as np
from mathutils import Vector
scene = bpy.context.scene
ob = bpy.context.object

def get_max(bound_box):
	max_x = max([bound_box[i][0] for i in range(0, 8)])
	max_y = max([bound_box[i][1] for i in range(0, 8)])
	max_z = max([bound_box[i][2] for i in range(0, 8)])
	return Vector((max_x, max_y, max_z))

def get_min(bound_box):
	min_x = min([bound_box[i][0] for i in range(0, 8)])
	min_y = min([bound_box[i][1] for i in range(0, 8)])
	min_z = min([bound_box[i][2] for i in range(0, 8)])
	return Vector((min_x, min_y, min_z))

def get_center(bound_box):
    center_x = (sum([bound_box[i][0] for i in range(0, 8)]))/8.0
    center_y = (sum([bound_box[i][1] for i in range(0, 8)]))/8.0
    center_z = (sum([bound_box[i][2] for i in range(0, 8)]))/8.0
    return Vector((center_x, center_y, center_z))

def get_max_distance(max_coord, min_coord):
    dist = (max_coord.x - min_coord.x)**2
    dist += (max_coord.y - min_coord.y)**2
    dist += (max_coord.z - min_coord.z)**2
    dist = np.sqrt(dist)
    return dist

# translation
'''
def translation(x, y, z):
    from mathutils import Matrix
    matrix = Matrix.Translation((x, y, z))
    mesh_unique = set(obj.data for obj in bpy.context.selected_objects)
    for mesh in mesh_unique:
        mesh.transform(matrix)
        mesh.update()
'''

def translation(x, y, z):
    meshes = bpy.data.meshes

    for mesh in meshes:
        for vertex in mesh.vertices:
            vertex.co += Vector((x, y, z))

'''
print(get_max(ob.bound_box))
print(get_min(ob.bound_box))
print(get_center(ob.bound_box))

'''

# 바운딩 박스의 모든 좌표 확인하기

'''
for i in range(0, 8):
    for j in range(0, 3):
        print(ob.bound_box[i][j])
    print('\n')

meshes = bpy.data.meshes
for mesh in meshes:
    for vertex in mesh.vertices:
        print(vertex.co)
    print('\n')
'''

'''
bounding_max_dist = get_max_distance(max_coord, min_coord)
print(bounding_max_dist)
'''
bounding_max_dist = 3.46
scale_default = 3.45 / bounding_max_dist # 잘 조절해보자

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=scale_default,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
  links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
  # Remap as other types can not represent the full range of depth.
  map = tree.nodes.new(type="CompositorNodeMapValue")
  # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
  map.offset = [-0.7]
  map.size = [args.depth_scale]
  map.use_min = True
  map.min = [0]
  links.new(render_layers.outputs['Depth'], map.inputs[0])

  links.new(map.outputs[0], depth_file_output.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'

# scale_normal.use_alpha = True
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
# bias_normal.use_alpha = True
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])


albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.obj)
for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

'''
# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
lamp.shadow_method = 'NOSHADOW' # RAY_SHADOW & NOSHADOW
# Possibly disable specular shading:
lamp.use_specular = False
'''
#scene = bpy.context.scene

###
# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
lamp_data.shadow_method = 'RAY_SHADOW'

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)

# Place lamp to a specified location
#lamp_object.location = (-0.52, 3, 0.914)
lamp_object.location = (-1.49, 2.11, 0.914)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object
###

# 보조 조명
'''
# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW' # 옵션 변경시켜도 큰 차이 X
lamp2.use_specular = False
lamp2.energy = 0.015
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180
'''


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin;
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


scene.render.resolution_x = 1000
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'

max_x = 0
min_x = 99999

max_y = 0
min_y = 99999

max_z = 0
min_z = 99999

meshes = bpy.data.meshes

# 바운딩 박스 큐브 생성을 위해 가장 큰/작은 x,y,z좌표 구하기
for mesh in meshes:
    for vertex in mesh.vertices:
        if(vertex.co.x > max_x):
            max_x = vertex.co.x
        if(vertex.co.x < min_x):
            min_x = vertex.co.x
        if(vertex.co.y > max_y):
            max_y = vertex.co.y
        if(vertex.co.y < min_y):
            min_y = vertex.co.y
        if(vertex.co.z > max_z):
            max_z = vertex.co.z
        if(vertex.co.z < min_z):
            min_z = vertex.co.z

#max_coord = get_max(ob.bound_box)
#min_coord = get_min(ob.bound_box)

max_coord = Vector((max_x, max_y, max_z))
min_coord = Vector((min_x, min_y, min_z))

bounding_max_dist = get_max_distance(max_coord, min_coord)
print("MAX_COORD")
print(max_coord)

print("MIN_COORD")
print(min_coord)

print("MAX_DIST")
print(bounding_max_dist)

# '(0,0,0) - bounding box의 중심'만큼 translation
#bound_box_center = get_center(ob.bound_box)
center_x = (max_x + min_x)/2
center_y = (max_y + min_y)/2
center_z = (max_z + min_z)/2
bound_box_center = Vector((center_x,center_y,center_z))
print("BOUND_BOX_CENTER")
print(bound_box_center)
translation(-(bound_box_center.x),-(bound_box_center.y),-(bound_box_center.z)) # scene 생성 뒤에 이동하지 않으면 반영 X

obj

#bpy.data.scenes["Scene"].render.bake_normal_space = 'TANGENT'
#scene.render.bake_normal_space = 'WORLD'

cam = scene.objects['Camera']
cam.location = (0, 1, 0.6)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
fp = os.path.join(args.output_folder, model_identifier, model_identifier)
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

#탄젠트냐 오브젝트냐 월드냐 기준으로
'''
탄젠트는.. 그 표면에 수직한 애의 상대 방향을 하겠다는거예요.
그래서 컬러피커를 해봤자 저 서피스의 탄젠트 방향에 대해서 구하고 있어을 수도 있겠다.. 
(탄젠트냐 오브젝트냐 월드냐 셋중 하난데.. 문제는 어쨌든 월드로 저장이 되어야 함)
이걸 잘 찾아보세요!

월드 방향으로 나와야 할거임
'''

from math import radians

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

for output_node in [depth_file_output, normal_file_output, albedo_file_output]:
    output_node.base_path = ''

for i in range(0, args.views):
    print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

    scene.render.filepath = fp + '_r_{0:03d}'.format(int(i * stepsize))
    depth_file_output.file_slots[0].path = scene.render.filepath + "_depth.png"
    normal_file_output.file_slots[0].path = scene.render.filepath + "_normal.png"
    albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo.png"

    bpy.ops.render.render(write_still=True)  # render still

    b_empty.rotation_euler[2] += radians(stepsize)
