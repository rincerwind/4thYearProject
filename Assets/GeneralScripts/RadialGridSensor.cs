using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;				// The farthest semi-circle of the sensor
	public int sectors;
	public float visionAngle;
	
	private Vector3 sensorOrigin;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);
	
	/*private void drawRedZoneRays(ref ArrayList rays){
		float sphereRadius = transform.parent.GetComponent<SphereCollider>().radius * transform.parent.localScale.x;

		// init ray origins
		Vector3 leftRayOrigin = sensorOrigin;
		leftRayOrigin.x -= sphereRadius;

		Vector3 rightRayOrigin = sensorOrigin;
		rightRayOrigin.x += sphereRadius;

		// init rays
		Ray leftRay = new Ray(leftRayOrigin, transform.rotation * Vector3.forward);
		Ray rightRay = new Ray(rightRayOrigin, transform.rotation * Vector3.forward);

		Debug.DrawRay(leftRay.origin, leftRay.direction * radius);
		Debug.DrawRay(rightRayOrigin, rightRay.direction * radius);

		rays.Add(leftRay);
		rays.Add(rightRay);
	}*/

	private void drawRays(ref ArrayList rays, 
	                            ref Vector3 rayDirection){
		//Debug.DrawRay(sensorOrigin, rayDirection * radius);
		Ray r1 = new Ray(sensorOrigin, rayDirection);
		rays.Add(r1);
	}
	
	public ArrayList collisionCheck(){
		Vector3 currRayDirection;
		
		ArrayList rays = new ArrayList();
		ArrayList hits = new ArrayList();

		// collide with everything that is not a Player
		int layerMask = ~(1<<LayerMask.NameToLayer("Player") | 1<<LayerMask.NameToLayer("Goal"));
		
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
			
			if( Physics.Raycast( curr, out hitInfo, radius, layerMask ) ){
				Debug.DrawRay( curr.origin, curr.direction * hitInfo.distance, Color.red );
				hits.Add (hitInfo.distance);
			}
			else
				hits.Add (0f);
		}
		return hits;
	}
}// end of class
