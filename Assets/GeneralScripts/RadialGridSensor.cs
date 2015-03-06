using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;				// The farthest semi-circle of the sensor
	public int sectors;
	public float visionAngle;
	
	private Vector3 sensorOrigin;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);
	private ArrayList rays;
	private int layerMask;

	private void drawRays(ref ArrayList rays, 
	                            ref Vector3 rayDirection){
		//Debug.DrawRay(sensorOrigin, rayDirection * radius);
		Ray r1 = new Ray(sensorOrigin, rayDirection);
		rays.Add(r1);
	}

	public float getRotAngle(){
		return visionAngle / sectors;
	}
	
	public ArrayList collisionCheck(){
		Vector3 currRayDirection;
		ArrayList hits = new ArrayList();
		rays = new ArrayList();

		// collide with everything that is NOT a Player or Goal
		layerMask = ~(1<<LayerMask.NameToLayer("Player") | 1<<LayerMask.NameToLayer("Goal"));
		
		// Init some variables
		sensorOrigin = new Vector3 (transform.position.x, 
		                            transform.position.y-0.3f, 
		                            transform.position.z);
		
		currRayDirection = transform.rotation * initRayDirection;
		float rot_angle = visionAngle / sectors;
		currRayDirection = Quaternion.AngleAxis( -(180 - visionAngle)/2, Vector3.up ) * currRayDirection;
		
		// Draw Radial Grid
		for( int curr_sector = 1; curr_sector <= sectors; curr_sector ++ ) {
			drawRays(ref rays, ref currRayDirection);
			currRayDirection = Quaternion.AngleAxis( -rot_angle, Vector3.up ) * currRayDirection;
			//drawRedZoneRays(ref rays);
		}
		
		drawRays (ref rays, ref currRayDirection);

		// Check for collisions
		RaycastHit hitInfo;

		for(int i = 0; i < rays.Count; i++){
			Ray curr = (Ray)rays[i];
			int mid = rays.Count/2;
			int length = (i != 0 && i != rays.Count - 1 && i != mid)? radius/2 : radius;
			
			if( Physics.Raycast( curr, out hitInfo, length, layerMask ) ){
				Debug.DrawRay( curr.origin, curr.direction * hitInfo.distance, Color.red );
				//hits.Add (hitInfo.distance);
				hits.Add (1f);
			}
			else{
				Debug.DrawRay( curr.origin, curr.direction * length, Color.white );
				hits.Add (0f);
			}
		}
		return hits;
	}
}// end of class
