import Sofa.Core
import Sofa.SofaDeformable

import numpy as np
from typing import Tuple, Optional, Union, Callable, List
from pathlib import Path
from sofa_env.sofa_templates.collision import COLLISION_PLUGIN_LIST, add_collision_model, CollisionModelType
from sofa_env.sofa_templates.deformable import DeformableObject, DEFORMABLE_PLUGIN_LIST
from sofa_env.sofa_templates.loader import add_loader, LOADER_PLUGIN_LIST
from sofa_env.sofa_templates.materials import Material, MATERIALS_PLUGIN_LIST
from sofa_env.sofa_templates.topology import TopologyTypes, TOPOLOGY_PLUGIN_LIST
from sofa_env.sofa_templates.solver import add_solver, ConstraintCorrectionType, SOLVER_PLUGIN_LIST
from sofa_env.sofa_templates.visual import add_visual_model, VISUAL_PLUGIN_LIST
from sofa_env.sofa_templates.scene_header import AnimationLoopType, SCENE_HEADER_PLUGIN_LIST
from sofa_env.sofa_templates.motion_restriction import MOTION_RESTRICTION_PLUGIN_LIST, add_bounding_box, add_rest_shape_spring_force_field_to_indices,add_fixed_constraint_in_bounding_box

INTESTINE_PLUGIN_LIST = [] + DEFORMABLE_PLUGIN_LIST + MATERIALS_PLUGIN_LIST + TOPOLOGY_PLUGIN_LIST + SOLVER_PLUGIN_LIST + VISUAL_PLUGIN_LIST + SCENE_HEADER_PLUGIN_LIST + LOADER_PLUGIN_LIST + COLLISION_PLUGIN_LIST


class Intestine(Sofa.Core.Controller):
    def __init__(
        self,
        parent_node: Sofa.Core.Node,
        volume_mesh_path: Union[str, Path],
        total_mass: float,
        visual_mesh_path: Union[str, Path],
        collision_mesh_path: Union[str, Path],
        name: str = "tissue",
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        volume_mesh_type: TopologyTypes = TopologyTypes.TETRA,
        material: Optional[Material] = None,
        add_solver_func: Callable = add_solver,
        add_visual_model_func: Callable = add_visual_model,
        animation_loop_type: AnimationLoopType = AnimationLoopType.DEFAULT,
        constraint_correction_type: ConstraintCorrectionType = ConstraintCorrectionType.PRECOMPUTED,
        show_object: bool = False,
        grasping_points_from: str = "visual",
    ):
        """Graspable tissue for tissue retraction.

        Args:
            parent_node (Sofa.Core.Node): Parent node of the object.
            name (str): Name of the object.
            volume_mesh_path (Union[str, Path]): Path to the volume mesh.
            total_mass (float): Total mass of the deformable object.
            visual_mesh_path (Optional[Union[str, Path]]): Path to the visual surface mesh.
            rotation (Tuple[float, float, float]): RPY rotations in degrees of the collision model in relation to the parent node. Order is X*Y*Z.
            translation (Tuple[float, float, float]): XYZ translation of the collision model in relation to the parent node.
            scale (float): Scale factor for loading the meshes.
            volume_mesh_type (TopologyTypes): Type of the volume mesh (e.g. TopologyTypes.TETRA for Tetraeder).
            material (Optional[Material]): Description of the material behavior.
            add_solver_func (Callable): Function that adds the numeric solvers to the object.
            add_visual_model_func (Callable): Function that defines how the visual surface from visual_mesh_path is added to the object.
            animation_loop_type (AnimationLoopType): The animation loop of the scene. Required to determine if objects for constraint correction should be added.
            constraint_correction_type (ConstraintCorrectionType): Type of constraint correction for the object, when animation_loop_type is AnimationLoopType.FREEMOTION.
            show_object (bool): Whether to render the nodes of the volume mesh.
            grasping_points_from (str): From which of the meshes ('volume' or 'visual') to select points used for describing grasping.
        """

        Sofa.Core.Controller.__init__(self)

        self.name = f"{name}_controller"

        self.deformable_object = DeformableObject(
            parent_node=parent_node,
            volume_mesh_path=volume_mesh_path,
            total_mass=total_mass,
            name=name,
            visual_mesh_path=visual_mesh_path,
            rotation=rotation,
            translation=translation,
            scale=scale,
            volume_mesh_type=volume_mesh_type,
            material=material,
            add_solver_func=add_solver_func,
            add_visual_model_func=add_visual_model_func,
            animation_loop_type=animation_loop_type,
            constraint_correction_type=constraint_correction_type,
            show_object=show_object,
            collision_mesh_path = collision_mesh_path
        )

        
        self.collision_spring_force_field_indices = [
            39,
            24,
            22,
            25,
            29,
            40,
            0,
            163,
            27,
            60,
            164,
            61,
            59,
            179,
            28,
            20,
            26,
            2,
            1,
            74,
            99,
            109,
            82,
            196,
        ]
        
        
        #l = [162,133,151,150,110,122,114,166,124,122,123,148,192,164,149,147,173,180,94,176,160,175,366,367]
        #l1 = [i-1 for i in l]
        #self.collision_spring_force_field_indices =l1
        self.restSpringsForceField = add_rest_shape_spring_force_field_to_indices(
            attached_to=self.deformable_object.collision_model_node,
            indices=self.collision_spring_force_field_indices,
            stiffness=1e5,
            angular_stiffness=1e5,
        )
       
        


        mesh_roi_node = self.deformable_object.node.addChild("MeshROI")

        #add_loader(attached_to=mesh_roi_node, file_path=liver_mesh_path, name="ROIloader", scale=scale)
        #print('it is half done ',self.deformable_object.node.MechanicalObject.position.getLinkPath())
        
        self.mesh_roi = mesh_roi_node.addObject(
            "MeshROI",
            computeMeshROI=True,
            doUpdate=True,
            position=self.deformable_object.node.MechanicalObject.position.getLinkPath(),
            ROIposition="@ROIloader.position",
            ROItriangles="@ROIloader.triangles",
        )
                

        self.graspable_region_box = add_bounding_box(
            attached_to=self.deformable_object.node,
            min=(-30, -30, -30),
            max=(30, 10, 10),
            name="grasp_box",
            extra_kwargs={"position": self.deformable_object.mechanical_object.position.getLinkPath()},
            rotation=(0, 0, -0),
        )

        self.graspable_region_box_collision = add_bounding_box(
            attached_to=self.deformable_object.collision_model_node,
            min=(-30, -30, -30),
            max=(30, 10, 10),
            name="grasp_box",
            extra_kwargs={"position": self.deformable_object.collision_model_node.MechanicalObject.position.getLinkPath()},
            rotation=(0, 0, -0),
        )
        

    def get_internal_force_magnitude(self) -> np.ndarray:
        """Get the sum of magnitudes of the internal forces applied to each vertex of the mesh"""
        return np.sum(np.linalg.norm(self.deformable_object.mechanical_object.force.array(), axis=1))
