from functools import partial
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Dict

import Sofa.Core

from sofa_env.sofa_templates.motion_restriction import add_bounding_box, MOTION_RESTRICTION_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import VISUAL_STYLES, AnimationLoopType, ConstraintSolverType, IntersectionMethod, add_scene_header, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.camera import Camera, CAMERA_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, ConstitutiveModel, MATERIALS_PLUGIN_LIST
from sofa_env.utils.camera import determine_look_at
from sofa_env.sofa_templates.solver import add_solver,SOLVER_PLUGIN_LIST, ConstraintCorrectionType
from sofa_env.scenes.soft_body_manipulation.sofa_objects.intestine1 import Intestine, INTESTINE_PLUGIN_LIST
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper1 import Gripper1, GRIPPER_PLUGIN_LIST
from sofa_env.scenes.soft_body_manipulation.sofa_objects.gripper2 import Gripper2, GRIPPER_PLUGIN_LIST
#from sofa_env.scenes.soft_body_manipulation.sofa_objects.tool_controller1 import ToolController
from sofa_env.sofa_templates.deformable import rigidify, DEFORMABLE_PLUGIN_LIST,add_fixed_constraint_in_bounding_box,add_fixed_constraint_to_indices

HERE = Path(__file__).resolve().parent
INSTRUMENT_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/instruments"
MODEL_MESH_DIR = HERE.parent.parent.parent / "assets/meshes/models"
tissue_translation = (0.02, 0.0, 0.045)
PLUGIN_LIST = (
    [
        "SofaPython3",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        'Sofa.Component.MechanicalLoad',
    ]
    + SCENE_HEADER_PLUGIN_LIST
    + INTESTINE_PLUGIN_LIST
    + GRIPPER_PLUGIN_LIST
    + CAMERA_PLUGIN_LIST
    + MOTION_RESTRICTION_PLUGIN_LIST
    + VISUAL_PLUGIN_LIST
    + MATERIALS_PLUGIN_LIST
)

GRAVITY = 9.8


def createScene(
    root_node: Sofa.Core.Node,
    image_shape: Tuple[Optional[int], Optional[int]] = (128, 128),
    show_everything: bool = False,
    animation_loop: AnimationLoopType =  AnimationLoopType.DEFAULT,
) -> Dict:

    """
    Creates the scene of the GraspLiftTouchEnv.

    Args:
        root_node (Sofa.Core.Node): SET BY ENV. The simulation tree's root node.
        image_shape (Tuple[Optional[int], Optional[int]]): SET BY ENV. Height and Width of the rendered images.
        show_everything (bool): Render additional information for debugging.
        animation_loop (AnimationLoopType): Animation loop of the simulation.

    Returns:
        scene_creation_result = {
            "interactive_objects": {"gripper": gripper, "cauter": cauter},
            "contact_listener": {
                "cauter_intestine": cauter_intestine_listener,
                "cauter_liver": cauter_liver_listener,
                "gripper_liver": gripper_liver_listener,
            },
            "poi": poi,
            "camera": camera,
            "liver": liver,
            "intestine": intestine,
        }
        182 151 148 150 123 122 124 166 124 173 123 147 149 164
    """

    add_scene_header(
        root_node=root_node,
        plugin_list=PLUGIN_LIST,
        visual_style_flags=VISUAL_STYLES["debug"] if show_everything else VISUAL_STYLES["normal"],
        gravity=(0.0, GRAVITY, 0.0),
        constraint_solver=ConstraintSolverType.GENERIC,
        constraint_solver_kwargs={
            "maxIt": 1000,
            "tolerance": 0.001,
        },
        animation_loop=animation_loop,
        scene_has_collisions=True,
        collision_detection_method_kwargs={
            "alarmDistance": 5.0,
            "contactDistance": 0.5,
        },
        collision_detection_method=IntersectionMethod.LOCALMIN,
    )

    #root_node.addObject("LightManager")

    camera = Camera(
        root_node,
        {
            "position": np.array([-94.5, -113.6, -237.2]),
            "orientation": np.array([0.683409, -0.684564, 0.0405591, 0.25036]),
            "lookAt": determine_look_at(np.array([-94.5, -113.6, -237.2]), np.array([0.683409, -0.684564, 0.0405591, 0.25036])),
        },
        z_near=100,
        z_far=500,
        width_viewport=image_shape[0],
        height_viewport=image_shape[1],
    )
    '''
    root_node.addObject(
        "DirectionalLight",
        direction=[94.5, 113.6, 237.2],
    )
    '''
    root_node.addObject(
        "LightManager",
        listening=True,
        ambient=(
            0.4,
            0.4,
            0.4,
            0.4,
        ),
    )

    root_node.addObject("PositionalLight", position=np.array([-94.5, -113.6, -237.2]))

    scene_node = root_node.addChild("scene")
    intestine_surface_texture_path = HERE / "meshes/liver_texture_2.png" 
    cartesian_instrument_workspace = {
        "low": np.array((-200, -200, -200)),
        "high": np.array((130, 130, 130)),
    }
    for array in cartesian_instrument_workspace.values():
        array.flags.writeable = False

    if show_everything:
        scene_node.addObject("MechanicalObject")
        add_bounding_box(
            scene_node,
            min=cartesian_instrument_workspace["low"],
            max=cartesian_instrument_workspace["high"],
            show_bounding_box=True,
            show_bounding_box_scale=4,
        )
        
    '''
    intestine = Intestine(
        parent_node=scene_node,
        volume_mesh_path=HERE / "meshes/livers/vh_m_liver_NIH3D_msh.msh",
        collision_mesh_path=HERE / "meshes/livers/vh_m_liver_NIH3D.stl",
        visual_mesh_path=HERE / "meshes/livers/vh_m_liver_NIH3D.stl",
        animation_loop=animation_loop
        )
    '''
    add_visual_intestine_model = partial(add_visual_model, texture_file_path=str(intestine_surface_texture_path))
    intestine =  scene_node.addObject(
        Intestine(
            animation_loop_type=animation_loop,
            constraint_correction_type=ConstraintCorrectionType.PRECOMPUTED,
            parent_node=scene_node,
            volume_mesh_path=HERE / "meshes/gallbladder_volumetric.vtk",
            collision_mesh_path = HERE / "meshes/gallbladder_collision.stl",
            total_mass=15.0,
            visual_mesh_path=HERE / "meshes/gallbladder_1.obj",
            translation=(0.0,0.0,0.0),
            rotation= (0.0,0.0,0.0),
            material=Material(poisson_ratio=0.0, young_modulus=1500),
            add_visual_model_func=add_visual_intestine_model,
            show_object=False,
        )
    )
    intestine.deformable_object.node.addObject(
            "UniformVelocityDampingForceField",
            template="Vec3d",
            name="Damping",
            dampingCoefficient=1,
        )
       
    
    
        
    #intestine.node.init()
    #intestine.init()
    #l = [277,263,264,265,545,275,281,282,288,419,296,299,295,293,280]
    #intestine.fix_indices(
    #    indices=[[i] for i in l],
    #    fixture_func=add_fixed_constraint_to_indices,
    #)
    # Rigidification of the elasticobject for given indices with given frameOrientations.
    #intestine.addObject("FixedConstraint", indices=[[i] for i in range(6805,6825)])\
    
    gripper1 = Gripper1(
        parent_node=scene_node,
        name="Gripper",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "single_action_forceps_shaft.stl",
        visual_mesh_path_jaw=INSTRUMENT_MESH_DIR / "single_action_forceps_jaw.stl",
        intestine_to_grasp=intestine,
        ptsd_state=np.array([-15.0, 30.0, -20.0, 130.0]),
        angle=45,
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        rcm_pose=np.array([-160.0, 30.0, -200.0, 0.0, 20.0, 45.0]),
        animation_loop_type=animation_loop,
        cartesian_workspace=cartesian_instrument_workspace,
        state_limits={
            "low": np.array([-75.0, -40.0, -1000.0, 12.0]),
            "high": np.array([75.0, 75.0, 1000.0, 300.0]),
        },
        total_mass=1e5,
    )
    scene_node.addObject(gripper1)
    '''
    gripper2 = Gripper2(
        parent_node=scene_node,
        name="Gripper2",
        visual_mesh_path_shaft=INSTRUMENT_MESH_DIR / "single_action_forceps_shaft.stl",
        visual_mesh_path_jaw=INSTRUMENT_MESH_DIR / "single_action_forceps_jaw.stl",
        intestine_to_grasp=intestine,
        ptsd_state=np.array([-15.0, 30.0, -20.0, 130.0]),
        angle=-45,
        collision_spheres_config={
            "positions": [[0, 0, 5 + i * 2] for i in range(10)],
            "radii": [1] * 10,
        },
        rcm_pose=np.array([120.0, 0.0, 0.0, 60.0, -30.0, -60.0]),
        animation_loop_type=animation_loop,
        cartesian_workspace=cartesian_instrument_workspace,
        state_limits={
            "low": np.array([-75.0, -40.0, -1000.0, 12.0]),
            "high": np.array([75.0, 75.0, 1000.0, 300.0]),
        },
        total_mass=1e5,
    )
    scene_node.addObject(gripper2)
    

    #scene_node.addObject(ToolController(gripper1=gripper1, gripper2=gripper2))
    
    '''

    gripper1_collision_model = gripper1.physical_shaft_node.approximated_collision_shaft_jaw_1.SphereCollisionModel.getLinkPath()
    #gripper2_collision_model = gripper2.physical_shaft_node.approximated_collision_shaft_jaw_2.SphereCollisionModel.getLinkPath()
    intestine_collision_model = intestine.deformable_object.collision_model_node.TriangleCollisionModel.getLinkPath()
    '''
    gripper2_intestine_listener = scene_node.addObject(
        "ContactListener",
        name="ContactListenerGripper1Intestine",
        collisionModel1=gripper2_collision_model,
        collisionModel2=intestine_collision_model,
    )
    '''


    gripper1_intestine_listener = scene_node.addObject(
        "ContactListener",
        name="ContactListenerGripper1Intestine",
        collisionModel1=gripper1_collision_model,
        collisionModel2=intestine_collision_model,
    )
    

    return {
        "interactive_objects": {"gripper1": gripper1},
        "contact_listener": {
            "gripper1_intestine": gripper1_intestine_listener,
        },
        "camera": camera,
        "intestine": intestine,
    }
      
