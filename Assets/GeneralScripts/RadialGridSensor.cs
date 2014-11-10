using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;				// The farthest semi-circle of the sensor
	public int sectors;				// Number of sectors in the sensor
	public int numSemiCircles;		// Number of semi-circles
	
	private Vector3 sensorOrigin;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);

	private void drawInnerRays(ref ArrayList rays, ref ArrayList magnitudes, ref Vector3 oldRayDir, ref Vector3 currRayDir, ref float depth){
		Vector3 innerRayDirection = currRayDir - oldRayDir;
		Vector3 oldRayEndPoint, currRayEndPoint;
		Ray innerRay = new Ray ();
		float currMagnitude;

		for (float innerRadius = radius; innerRadius > 0; innerRadius -= depth) {
			oldRayEndPoint = sensorOrigin + oldRayDir.normalized * innerRadius;
			currRayEndPoint = sensorOrigin + currRayDir.normalized * innerRadius;
			currMagnitude = Vector3.Distance(oldRayEndPoint, currRayEndPoint);

			innerRay = new Ray(oldRayEndPoint, 
			                   innerRayDirection);
			Debug.DrawRay (innerRay.origin, innerRay.direction * currMagnitude);

			rays.Add(innerRay);
			magnitudes.Add(currMagnitude);
		}
	}

	private void drawOutterRays(ref ArrayList rays, ref ArrayList magnitudes, ref Vector3 rayDirection, ref float depth){
		Ray outterRay = new Ray ();

		for (int i = 0; i * depth <= radius; i++) {
			outterRay = new Ray(sensorOrigin + rayDirection.normalized * i * depth, 
			                    rayDirection);
			Debug.DrawRay (outterRay.origin, outterRay.direction * depth);

			rays.Add(outterRay);
			magnitudes.Add(depth);
		}
	}

	void Update(){
		Vector3 currRayDirection;
		Vector3 oldRayDirection;
		ArrayList rays = new ArrayList();
		ArrayList magnitudes = new ArrayList();
		RaycastHit hitInfo;
		
		float depth = (float)radius / numSemiCircles;
		sensorOrigin = new Vector3 (transform.position.x, 
		                            transform.position.y, 
		                            transform.position.z);
		
		currRayDirection = transform.rotation * initRayDirection;
		float rot_angle = 180f / sectors;
		
		for( int curr_sector = 1; curr_sector <= sectors; curr_sector ++ ) {
			drawOutterRays(ref rays, ref magnitudes, ref currRayDirection, ref depth);

			oldRayDirection = currRayDirection;
			currRayDirection = Quaternion.AngleAxis( -rot_angle, Vector3.up ) * currRayDirection;

			drawInnerRays(ref rays, ref magnitudes, ref oldRayDirection, ref currRayDirection, ref depth);
		}
		
		drawOutterRays (ref rays, ref magnitudes, ref currRayDirection, ref depth);

		for (int i = 0; i < rays.Count; i++) {
			Ray currRay = (Ray)rays[i];

			if (Physics.Raycast (currRay, out hitInfo, (float)magnitudes[i]) )
				Debug.DrawRay (currRay.origin, currRay.direction * (float)magnitudes[i], Color.red);
		}
	}// end of Update
}// end of class
