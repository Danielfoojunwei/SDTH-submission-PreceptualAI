"""Real Blender scene models for the concise deck. Run with Blender 4.3+.

These are explicitly schematic 3D illustrations. The radio-map mesh uses an
actual predicted grid; arrows/halos encode relationships, not measured rays.
"""
import bpy
import json,math,random,sys,argparse
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT=Path(__file__).resolve().parents[3]
PALETTE={'navy':(0.012,.027,.045),'floor':(.025,.055,.075),'steel':(.045,.09,.13),
 'white':(.79,.85,.84),'teal':(.025,.66,.58),'cyan':(.12,.83,.83),
 'orange':(.96,.34,.105),'cream':(.76,.63,.45),'red':(.75,.07,.055)}
CACHE={};MATS={};MARKERS={}


def material(name,color,metal=.0,rough=.45,emission=0.,alpha=1.,transmission=0.):
    m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
    p.inputs['Alpha'].default_value=alpha;p.inputs['Transmission Weight'].default_value=transmission
    if emission:p.inputs['Emission Color'].default_value=(*color,1);p.inputs['Emission Strength'].default_value=emission
    return m


def init_materials():
    global MATS
    MATS={n:material(n,c,metal=.3 if n in ['steel','floor','white'] else .1,rough=.35) for n,c in PALETTE.items()}
    for n in ['teal','cyan','orange','red']:
        MATS[n+'_glow']=material(n+'_glow',PALETTE[n],rough=.3,emission=3.)
    MATS['glass']=material('glass',PALETTE['cyan'],rough=.22,metal=.05,alpha=.3,transmission=.3)
    MATS['glass_orange']=material('glass_orange',PALETTE['orange'],rough=.25,alpha=.36,transmission=.2)


def group(name,loc=(0,0,0),scale=1.):
    obj=bpy.data.objects.new(name,None);bpy.context.collection.objects.link(obj);obj.location=loc;obj.scale=(scale,)*3;return obj


def finish(obj,name,loc,mat,parent=None):
    obj.name=name;obj.location=loc
    if parent:obj.parent=parent
    if mat:obj.data.materials.clear();obj.data.materials.append(MATS[mat] if isinstance(mat,str) else mat)
    return obj


def box(name,loc,dims,mat='steel',bevel=.06,parent=None):
    key=('box',tuple(round(x,5) for x in dims),mat.name if hasattr(mat,'name') else mat,bevel)
    if key not in CACHE:
        bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,0));o=bpy.context.object;o.scale=dims
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        if bevel:
            mod=o.modifiers.new('Machined bevel','BEVEL');mod.width=min(bevel,min(dims)*.22);mod.segments=3
            bpy.ops.object.modifier_apply(modifier=mod.name)
            mod=o.modifiers.new('Weighted normals','WEIGHTED_NORMAL');mod.keep_sharp=True
            bpy.ops.object.modifier_apply(modifier=mod.name)
        mesh=o.data;CACHE[key]=mesh;bpy.data.objects.remove(o,do_unlink=True)
    obj=bpy.data.objects.new(name,CACHE[key]);bpy.context.collection.objects.link(obj)
    return finish(obj,name,loc,mat,parent)


def sphere(name,loc,r=.1,mat='teal_glow',parent=None):
    key=('sphere',r,mat)
    if key not in CACHE:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=r)
        o=bpy.context.object
        for p in o.data.polygons:p.use_smooth=True
        CACHE[key]=o.data;bpy.data.objects.remove(o,do_unlink=True)
    o=bpy.data.objects.new(name,CACHE[key]);bpy.context.collection.objects.link(o);return finish(o,name,loc,mat,parent)


def cylinder(name,loc,radius,depth,mat='white',parent=None,rotation=None):
    key=('cylinder',radius,depth,mat)
    if key not in CACHE:
        bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=radius,depth=depth)
        o=bpy.context.object
        for p in o.data.polygons:p.use_smooth=True
        CACHE[key]=o.data;bpy.data.objects.remove(o,do_unlink=True)
    o=bpy.data.objects.new(name,CACHE[key]);bpy.context.collection.objects.link(o);finish(o,name,loc,mat,parent)
    if rotation:o.rotation_euler=rotation
    return o


def ring(name,loc,radius,mat='teal_glow',parent=None,thickness=.02):
    pts=[(loc[0]+radius*math.cos(i*math.tau/80),loc[1]+radius*math.sin(i*math.tau/80),loc[2]) for i in range(81)]
    return tube(name,pts,thickness,mat,parent)


def tube(name,points,width=.035,mat='teal_glow',parent=None):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=12;curve.bevel_depth=width;curve.bevel_resolution=3
    spline=curve.splines.new('POLY');spline.points.add(len(points)-1)
    for p,xyz in zip(spline.points,points):p.co=(*xyz,1)
    o=bpy.data.objects.new(name,curve);bpy.context.collection.objects.link(o)
    if parent:o.parent=parent
    curve.materials.append(MATS[mat]);return o


def arc(name,a,b,height=1.,mat='teal_glow',parent=None,width=.035,beads=False):
    points=[]
    for i in range(41):
        t=i/40;v=Vector(a).lerp(Vector(b),t);v.z+=math.sin(math.pi*t)*height;points.append(tuple(v))
    tube(name,points,width,mat,parent)
    if beads:
        for i in [8,18,28]:sphere(name+'_packet',points[i],width*2.8,mat,parent)


def marker(name,point,parent=None):
    o=group(name,point)
    if parent:o.parent=parent
    MARKERS[name]=o


def robot(name,point,parent=None,scale=1.,focus=False):
    g=group(name,point,scale)
    if parent:g.parent=parent
    box(name+'_base',(0,0,.16),(.48,.36,.21),'white',.07,g)
    box(name+'_deck',(0,0,.29),(.37,.27,.06),'steel',.025,g)
    box(name+'_strip',(0,-.185,.17),(.31,.012,.065),'orange_glow' if focus else 'teal_glow',.005,g)
    for x in [-.15,.15]:
        for y in [-.2,.2]:cylinder(name+'_wheel',(x,y,.115),.095,.065,'navy',g,(math.pi/2,0,0))
    cylinder(name+'_sensor',(.11,0,.36),.065,.09,'teal_glow',g)
    for x in [-.1,0.0]:cylinder(name+'_antenna',(x,.06,.41),.012,.25,'steel',g)
    return g


def ap(name,point,parent=None,kind='wifi',height=1.7):
    g=group(name,point)
    if parent:g.parent=parent
    cylinder(name+'_mast',(0,0,height/2),.025,height,'steel',g)
    cylinder(name+'_base',(0,0,.06),.24,.08,'floor',g)
    if kind=='wifi':
        box(name+'_head',(0,0,height),(.36,.36,.11),'white',.06,g)
        ring(name+'_halo',(0,0,height+.07),.14,'teal_glow',g,.014)
    else:
        cylinder(name+'_panel',(0,0,height-.05),.12,.65,'white',g)
        for z in [height-.25,height,height+.25]:ring(name+'_band',(0,0,z),.135,'orange_glow',g,.012)
    return g


def warehouse(data,name='warehouse',loc=(0,0,0),scale=1.,robots=28,sites=True,racks=True):
    g=group(name,loc,scale)
    box(name+'_slab',(0,0,-.18),(14.8,10.,.38),'floor',.17,g)
    for x in [-7.15,7.15]:tube(name+'_edge',[(x,-4.7,.03),(x,4.7,.03)],.018,'teal_glow',g)
    for y in [-4.65,4.65]:tube(name+'_edge',[(-7.15,y,.03),(7.15,y,.03)],.018,'teal_glow',g)
    if racks:
        for row,y in enumerate(data['rack_rows']):
            for j,(a,b) in enumerate(data['rack_segments']):
                width=(b-a)*.12;cx=((a+b)/2-60)*.12;cy=(y-40)*.12
                for z in [.13,.49,.85]:box('shelf',(cx,cy,z),(width,.21,.06),'steel',.018,g)
                for x in [cx-width/2+.04,cx+width/2-.04]:box('upright',(x,cy,.48),(.055,.24,.97),'white',.012,g)
                for p in range(3):
                    x=cx-width*.28+p*width*.28
                    box('stock',(x,cy,.66),(.38,.2,.24),'cream',.025,g)
    if sites:
        for i,(x,y) in enumerate(data['sites']):ap('AP_'+str(i),((x-60)*.12,(y-40)*.12,0),g,height=1.8)
    for i,(x,y) in enumerate(data['robots'][:robots]):robot('AMR_'+str(i),((x-60)*.12,(y-40)*.12,.03),g,scale=.8,focus=i==16)
    return g


def plate(name,loc,dims,color='glass',parent=None):return box(name,loc,(*dims,.12),color,.1,parent)


def label3d(name,value,loc,size=.38,mat='white',parent=None):
    c=bpy.data.curves.new(name,'FONT');c.body=value;c.align_x='CENTER';c.size=size;c.extrude=.002
    obj=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(obj);obj.location=loc;obj.rotation_euler=(math.pi/2,0,0)
    if parent:obj.parent=parent
    c.materials.append(MATS[mat]);return obj


def field_surface(data,parent=None,height=1.4,scale=1.):
    grid=data['map'];xs=grid['x'];ys=grid['y'];values=grid['mean_dbm'];verts=[];cols=[]
    for j,row in enumerate(xs):
        for i,x in enumerate(row):
            v=max(0.,min(1.,(values[j][i]+95)/55));verts.append(((x-60)*.12*scale,(ys[j][i]-40)*.12*scale,height+v*2.8*scale));cols.append((.015+v*.8,.28+v*.32,.4-v*.18,1))
    n=len(xs[0]);faces=[]
    for j in range(len(xs)-1):
        for i in range(n-1):a=j*n+i;faces.append((a,a+1,a+n+1,a+n))
    mesh=bpy.data.meshes.new('Actual RadioMap RSSI grid');mesh.from_pydata(verts,[],faces);mesh.update()
    attr=mesh.color_attributes.new(name='signal',type='FLOAT_COLOR',domain='POINT')
    for a,c in zip(attr.data,cols):a.color=c
    m=material('RSSI surface',(.05,.5,.55),rough=.32,metal=.35,alpha=.66)
    nodes=m.node_tree.nodes;p=nodes.get('Principled BSDF');a=nodes.new('ShaderNodeVertexColor');a.layer_name='signal';m.node_tree.links.new(a.outputs['Color'],p.inputs['Base Color']);m.node_tree.links.new(a.outputs['Color'],p.inputs['Emission Color']);p.inputs['Emission Strength'].default_value=.25
    o=bpy.data.objects.new('RSSI surface',mesh);bpy.context.collection.objects.link(o);mesh.materials.append(m)
    if parent:o.parent=parent
    for j in range(0,len(xs),4):tube('surface row',[(x,y,z+.02) for x,y,z in verts[j*n:(j+1)*n]],.009,'cyan_glow',parent)
    for i in range(0,n,4):tube('surface col',[(verts[j*n+i][0],verts[j*n+i][1],verts[j*n+i][2]+.02) for j in range(len(xs))],.009,'cyan_glow',parent)
    return o


def scene_warehouse(data):
    g=warehouse(data,robots=100)
    selected=[(-5,-2,.4),(0,1,.4),(4,-1,.4)]
    for i,p in enumerate(selected):
        arc('fleet link',p,(-3+i*4,3,1.9),2.0,'teal_glow',g,width=.025,beads=True)
    ring('coverage halo',(-4,2,.06),1.8,parent=g,thickness=.025)
    ring('coverage halo2',(3,-1,.06),2.,parent=g,thickness=.025)
    marker('system',(1,0,2),g)


def scene_architecture(data):
    g=warehouse(data,scale=.75,robots=16,sites=False)
    plate('Map layer',(0,0,2.0),(11.0,7.5),'glass')
    for x in [-4,-2,0,2,4]:
        for y in [-2,0,2]:sphere('map state',(x,y,2.25),.11,'teal_glow')
    plate('Decision layer',(0,.35,4.25),(8.8,5.4),'glass_orange')
    for x in [-2.8,0,2.8]:
        box('Policy module',(x,.35,4.58),(1.6,2.,.5),'white',.13)
        box('Policy core',(x,.35,4.86),(1.0,1.4,.08),'orange_glow',.03)
    for a,b in [((-4,-2,.2),(-4,-2,2.2)),((4,2,2.2),(2.8,.35,4.8))]:arc('upstream',a,b,.4,'cyan_glow',width=.045,beads=True)
    arc('closed loop',(3.6,.4,4.8),(4.5,-3.4,.2),2.,'orange_glow',width=.04,beads=True)
    marker('learn',(-4,1,2.3));marker('decide',(0,.4,4.9));marker('recover',(4.5,-3.5,.4))


def scene_radiomap(data):
    warehouse(data,robots=12,sites=False)
    field_surface(data,height=1.0)
    ap('Observed site',(-1.8,1.4,0),height=2.4)
    for r in [1.4,2.8,4.2]:ring('inferred contour',(-1.8,1.4,.08),r,'teal_glow',thickness=.025)
    marker('map_peak',(-1.8,1.4,4.0))


def scene_mission(data):
    plate('Mission console',(0,0,.2),(12.,7.),'floor')
    robot('Priority robot',(1.8,-2.1,.35),scale=3.,focus=True)
    for i in range(6):
        box('Mission token',(-4+i*.42,2.2,.65),(.3,.8,.25),'orange_glow',.04)
    for i,(x,y) in enumerate([(-2,1.7),(1,2.2),(4.2,1.1)]):
        cylinder('Control dial',(x,y,.6),.86,.28,'white');ring('Control ring',(x,y,.78),.65,'orange_glow',thickness=.065)
        arc('intent compiler',(-3.2,2.2,.8),(x,y,.9),1.3,'orange_glow',width=.035,beads=True)
        arc('policy action',(x,y,.9),(1.8,-2.1,1.4),1.1,'teal_glow',width=.035)
    marker('focus_robot',(1.8,-2.1,1.8));marker('knobs',(1,2.2,1.4))


def scene_gate(data):
    box('Gate base',(0,0,-.25),(12.,7.2,.4),'floor',.2)
    for x in [-.65,.65]:box('Confidence pillar',(x,.7,1.8),(.27,.55,3.8),'white',.1)
    box('Confidence lintel',(0,.7,3.7),(1.7,.65,.35),'white',.12)
    box('Confidence volume',(0,.7,1.95),(1.2,.3,3.2),'glass',.06)
    arc('confident action',(-5,0,.25),(4,1.2,.3),1.2,'orange_glow',width=.08,beads=True)
    arc('fallback action',(-5,0,.25),(3.6,-2.7,.3),.5,'teal_glow',width=.06,beads=True)
    for i in range(7):sphere('uncertainty',(-3+i*.35,.7+(i%3)*.35,1.1+(i%2)*.5),.2,'glass')
    cylinder('Planned action terminal',(4,1.2,.2),.55,.25,'orange');cylinder('Reactive terminal',(3.6,-2.7,.2),.55,.25,'teal')
    marker('planned',(4,1.2,.7));marker('reactive',(3.6,-2.7,.7))


def scene_diversity(data):
    box('Aisle',(0,0,-.2),(13.,8.,.4),'floor',.2)
    for x in [-5,5]:
        for y in [-2.5,1.]:box('Rack',(x,y,.85),(2.,2.1,1.7),'steel',.1)
    r=robot('Dual-radio robot',(1,-2.2,.08),scale=3.2,focus=True)
    ap('AP A',(-4,3,.05),height=3.1);ap('AP B',(4,3,.05),height=3.1)
    box('Occluding stock',(-1.6,.2,1.2),(2.2,1.,2.4),'cream',.12)
    arc('blocked path',(1,-2.2,1),(-4,3,3.1),.3,'red_glow',width=.045)
    arc('independent path',(1,-2.2,1),(4,3,3.1),1.5,'teal_glow',width=.065,beads=True)
    for p in [(-1.9,.6,2.6),(-1.5,.6,2.6)]:sphere('Blockage flag',p,.12,'red_glow')
    marker('ap_a',(-4,3,3.5));marker('ap_b',(4,3,3.5));marker('robot',(1,-2.2,1.8))


def scene_stack(data):
    box('Stack stage',(0,0,-.25),(10.8,8.,.4),'floor',.18)
    robot('Radio platform',(0,-1.4,.1),scale=4.,focus=True)
    levels=[(1.35,'white'),(2.6,'glass'),(3.85,'white'),(5.1,'glass_orange')]
    for i,(z,mat) in enumerate(levels):
        plate('Layer '+str(i),(0,.4,z),(5.7,3.6),mat)
        for k in range(4):box('Traffic token',(-1.8+k*1.1,.4,z+.2),(.55,.7,.28),'orange_glow' if i==3 else 'teal_glow',.04)
        marker('layer_'+str(i),(2.8,.4,z+.2))
    tube('Control spine',[(-2.8,-1,1),(-2.8,-1,5.4)],.035,'cyan_glow')
    arc('Recovery loop',(-2.8,-1,5.2),(1,-1.6,.8),2.,'orange_glow',width=.03,beads=True)


def scene_result(data):
    for x,kind in [(-5.2,'wifi'),(5.2,'cellular')]:
        g=warehouse(data,'comparison_'+kind,(x,0,.0),scale=.5,robots=22,sites=False);g.rotation_euler.z=-.12
        ap('Radio 1',(-3,2,0),g,kind='wifi' if kind=='wifi' else 'cellular',height=2.3)
        if kind=='wifi':
            ap('Radio 2',(3,2,0),g,height=2.3)
            arc('Path A',(0,-2,.5),(-3,2,2.3),2.2,'teal_glow',g,width=.06,beads=True)
            arc('Path B',(0,-2,.5),(3,2,2.3),2.2,'teal_glow',g,width=.06,beads=True)
        else:
            for i in range(3):arc('Scheduled cell',(-2+i*2,-2,.4),(-3,2,2.3),1.5,'orange_glow',g,width=.045)
        marker(kind,(x,0,2.2))


def scene_envelope(data):
    for i,(x,condition) in enumerate([(-7.8,'clean'),(0,'main'),(7.8,'stress')]):
        g=warehouse(data,'condition_'+condition,(x,0,0),scale=.43,robots=14,sites=False);g.rotation_euler.z=-.12
        ap('AP_'+condition,(-2.7,2,0),g,height=2.4);ap('Path_B_'+condition,(2.7,2,0),g,height=2.4)
        for k in range(i+1):ring('RF occupancy',(-1.5+k*1.5,.0,1.9),1.5,'orange_glow' if i==2 else 'teal_glow',g,.03)
        if i==2:
            for j in range(5):arc('Noise',(0,-2,.3),(2.5-j,2,2.2),1.8,'red_glow',g,width=.04)
        marker(condition,(x,0,1.5))


def scene_validation(data):
    box('Test aisle',(0,0,-.2),(14.,8.,.4),'floor',.2)
    for i,x in enumerate([-4.8,-1.5,1.8,5.1]):
        for y in [-1.3,1.3]:box('Validation arch',(x,y,1.1),(.13,.17,2.4),'white',.035)
        box('Validation beam',(x,0,2.3),(.17,2.8,.14),'orange_glow' if i==3 else 'teal_glow',.025)
        marker('stage_'+str(i),(x,0,2.7))
    for i in range(10):robot('Pilot robot',(-5+(i%5)*2.4,-2.8+(i//5)*1.1,.05),scale=1.15,focus=i==0)
    for x,y in [(-5,3),(5,3),(-5,-3),(5,-3)]:ap('Pilot AP',(x,y,0),height=2.1)
    box('Measurement console',(4.5,2.,.85),(1.2,.65,1.7),'steel',.1)
    box('Console screen',(4.5,1.65,1.3),(.95,.05,.55),'teal_glow',.025)
    tube('Pilot route',[(-6,0,.04),(6,0,.04)],.04,'orange_glow')


BUILDERS={k:globals()['scene_'+k] for k in ['warehouse','architecture','radiomap','mission','gate','diversity','stack','result','envelope','validation']}


def frame_model(scene,name):
    """Fit actual model bounds, leaving deliberate space for editable typography."""
    objects=[o for o in scene.objects if o.type in {'MESH','CURVE'}]
    def bounds():
        points=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(corner))
                for o in objects for corner in o.bound_box]
        return min(p.x for p in points),max(p.x for p in points),min(p.y for p in points),max(p.y for p in points)
    wide=name in {'result','envelope'}
    width,height=(.91,.32) if wide else (.64,.73)
    x0,x1,y0,y1=bounds();scene.camera.data.ortho_scale*=max((x1-x0)/width,(y1-y0)/height)
    bpy.context.view_layer.update();x0,x1,y0,y1=bounds()
    scene.camera.data.shift_x+=(x0+x1)/2-(.5 if wide else .675)
    scene.camera.data.shift_y+=((y0+y1)/2-(.51 if wide else .50))*9/16
    bpy.context.view_layer.update()


def setup(name,width,samples):
    global CACHE,MARKERS
    bpy.ops.wm.read_factory_settings(use_empty=True);CACHE={};MARKERS={};init_materials()
    bpy.context.preferences.filepaths.save_version=0
    s=bpy.context.scene;s.name=name;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=samples;s.cycles.use_denoising=False
    s.cycles.max_bounces=5;s.cycles.diffuse_bounces=2;s.cycles.glossy_bounces=2;s.cycles.transmission_bounces=4
    s.render.threads_mode='FIXED';s.render.threads=4;s.render.resolution_x=width;s.render.resolution_y=int(width*9/16);s.render.resolution_percentage=100
    s.render.film_transparent=True;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA';s.render.image_settings.color_depth='8'
    s.world=bpy.data.worlds.new('Studio environment');s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs[0].default_value=(.06,.11,.18,1);s.world.node_tree.nodes['Background'].inputs[1].default_value=.55
    for loc,power,size,col in [((1,-8,16),2600,9,(.78,.89,1.)),((-10,2,10),1800,8,(.1,1.,.85)),((8,6,12),2400,7,(1.,.45,.2))]:
        bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.data.color=col
        o.rotation_euler=(Vector((0,0,1.5))-o.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add(location=(15,-24,20));cam=bpy.context.object;target=Vector((0,0,1.5));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=24;cam.data.shift_x=-.18;s.camera=cam
    if name in ['result','envelope']:
        cam.data.shift_x=0.;cam.data.ortho_scale=27 if name=='result' else 32;cam.location=(0,-30,23);cam.rotation_euler=(Vector((0,0,1))-cam.location).to_track_quat('-Z','Y').to_euler()
    if name in ['gate','mission']:cam.data.ortho_scale=20
    if name=='stack':cam.data.ortho_scale=20;cam.location=(14,-24,16);cam.rotation_euler=(Vector((0,0,2.5))-cam.location).to_track_quat('-Z','Y').to_euler()
    s.view_settings.view_transform='AgX';s.view_settings.look='AgX - Medium High Contrast'
    s.use_nodes=True;tree=s.node_tree;tree.nodes.clear();inp=tree.nodes.new('CompositorNodeRLayers');glow=tree.nodes.new('CompositorNodeGlare');glow.glare_type='FOG_GLOW';glow.quality='HIGH';glow.threshold=1.3;glow.size=7;out=tree.nodes.new('CompositorNodeComposite');tree.links.new(inp.outputs['Image'],glow.inputs['Image']);tree.links.new(glow.outputs['Image'],out.inputs['Image'])
    return s


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser();parser.add_argument('--scene',choices=list(BUILDERS)+['all'],default='all');parser.add_argument('--width',type=int,default=2304);parser.add_argument('--samples',type=int,default=48);parser.add_argument('--out',type=Path,default=ROOT/'docs/slides/concise/generated');args=parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True);data=json.loads((ROOT/'docs/slides/concise/assets/scene_data.json').read_text())
    names=list(BUILDERS) if args.scene=='all' else [args.scene]
    for name in names:
        s=setup(name,args.width,args.samples);BUILDERS[name](data);bpy.context.view_layer.update();frame_model(s,name)
        metadata={'scene':name,'blender_version':bpy.app.version_string,'schematic_3d_model':True,'measured_hardware':False,'width':args.width,'height':s.render.resolution_y,'samples':args.samples,'objects':len(s.objects),'mesh_objects':sum(o.type=='MESH' for o in s.objects),'markers':{}}
        for key,o in MARKERS.items():
            p=world_to_camera_view(s,s.camera,o.matrix_world.translation);metadata['markers'][key]=[float(p.x),float(1-p.y)]
        (args.out/f'{name}.model.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if name in ['architecture','warehouse','diversity']:
            models=args.out/'models';models.mkdir(exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(models/f'{name}.blend'))
            bpy.ops.export_scene.gltf(filepath=str(models/f'{name}.glb'),export_format='GLB',export_cameras=False,export_lights=False,export_apply=True)
        s.render.filepath=str(args.out/f'{name}.png');bpy.ops.render.render(write_still=True)
        print('VISUAL_DECK_RENDERED',name,flush=True)


if __name__=='__main__':main()
