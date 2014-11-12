using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;				// The farthest semi-circle of the sensor
	public int sectors;				// Number of sectors in the sensor
	public int numSemiCircles;		// Number of semi-circles
	
	private Vector3 sensorOrigin;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);
	//private ArrayList startPoints;
	//private ArrayList endPoints;

	private void drawInnerRays(ref ArrayList startPoints, ref ArrayList endPoints, 
	                           ref Vector3 oldRayDir, ref Vector3 currRayDir, ref float depth){
		Vector3 oldRayEndPoint, currRayEndPoint;

		for (float innerRadius = radius; innerRadius > 0; innerRadius -= depth) {
			oldRayEndPoint = sensorOrigin + oldRayDir.normalized * innerRadius;
			currRayEndPoint = sensorOrigin + currRayDir.normalized * innerRadius;

			Debug.DrawLine( oldRayEndPoint, currRayEndPoint );

			startPoints.Add(oldRayEndPoint);
			endPoints.Add(currRayEndPoint);
		}
	}

	private void drawOutterRays(ref ArrayList rays, ref ArrayList endPoints, 
	                            ref Vector3 rayDirection, ref float depth){
		/*Vector3 start, end;

		for (int i = 0; i * depth < radius; i++) {
			start = sensorOrigin + rayDirection.normalized * i * depth;
			end = start + rayDirection.normalized * depth;

			Debug.DrawLine(start, end);

			startPoints.Add(start);
			endPoints.Add(end);
		}*/
		Debug.DrawRay(sensorOrigin, rayDirection * radius);
		Ray r1 = new Ray(sensorOrigin, rayDirection);
		rays.Add(r1);
	}

	void Update(){
		Vector3 currRayDirection;
		Vector3 oldRayDirection;
		ArrayList startPoints = new ArrayList();
		ArrayList endPoints = new ArrayList();
		ArrayList rays = new ArrayList();
		RaycastHit hitInfo;
		int layerMask = ~(1<<LayerMask.NameToLayer("Player"));

		// Init some variables
		float depth = (float)radius / numSemiCircles;
		sensorOrigin = new Vector3 (transform.position.x, 
		                            transform.position.y-0.3f, 
		                           	transform.position.z);
		
		currRayDirection = transform.rotation * initRayDirection;
		float rot_angle = 180f / sectors;

		// Draw Radial Grid
		for( int curr_sector = 1; curr_sector <= sectors; curr_sector ++ ) {
			drawOutterRays(ref rays, ref endPoints, ref currRayDirection, ref depth);

			oldRayDirection = currRayDirection;
			currRayDirection = Quaternion.AngleAxis( -rot_angle, Vector3.up ) * currRayDirection;

			//drawInnerRays(ref startPoints, ref endPoints, ref oldRayDirection, ref currRayDirection, ref depth);
		}

		drawOutterRays (ref rays, ref endPoints, ref currRayDirection, ref depth);

		// Check rays for collisions
		for(int i = 0; i < rays.Count; i++){
			Ray curr = (Ray)rays[i];
		
			if( Physics.Raycast( curr, out hitInfo, radius, layerMask ) )
				Debug.DrawRay( curr.origin, curr.direction * hitInfo.distance, Color.red );
		}

		// Check lines for collisions
		for (int i = 0; i < startPoints.Count; i++) {
			Vector3 start = (Vector3)startPoints[i];
			Vector3 end = (Vector3)endPoints[i];

			if( Physics.Linecast( start, end, out hitInfo, layerMask )
			|| Physics.Linecast( end, start, out hitInfo, layerMask ) ){
				//print ("Hit!");
				Debug.DrawLine( start, end, Color.red );
			}
		}
	}// end of Update
}// end of class
