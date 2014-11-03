using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;				// The farthest semi-circle of the sensor
	public int sectors;				// Number of sectors in the sensor
	public int numSemiCircles;		// Number of semi-circles
	
	private Vector3 sensorOrigin;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);
	private ArrayList rays;

	private void drawInnerRays(Vector3 oldRayDir, Vector3 currRayDir, float depth){
		Vector3 innerRayDirection = currRayDir - oldRayDir;

		for (float innerRadius = radius; innerRadius > 0; innerRadius -= depth) {
			Debug.DrawRay (sensorOrigin + oldRayDir.normalized * innerRadius, innerRayDirection * innerRadius);
		}
	}

	void Update(){
		Vector3 currRayDirection;
		Vector3 oldRayDirection;

		float depth = (float)radius / numSemiCircles;
		sensorOrigin = new Vector3 (transform.position.x, 
		                            transform.position.y, 
		                            transform.position.z);

		currRayDirection = transform.rotation * initRayDirection;
		RaycastHit hitInfo;
		float rot_angle = 180f / sectors;
		
		for( float angle = 0f; angle < 180; angle += rot_angle ) {
			Debug.DrawRay (sensorOrigin, currRayDirection * radius);
			oldRayDirection = currRayDirection;
			currRayDirection = Quaternion.AngleAxis( -rot_angle, Vector3.up ) * currRayDirection;
			drawInnerRays(oldRayDirection, currRayDirection, depth);
		}

		Debug.DrawRay (sensorOrigin, currRayDirection * radius);
	}// end of Update
}// end of class
