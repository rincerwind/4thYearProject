using UnityEngine;
using System.Collections;

public class RadialGridSensor : MonoBehaviour {
	public int radius;	// The farthest semi-circle of the sensor
	public int sectors;	// Number of sectors in the sensor
	public int depth;	// Number of semi-circles

	private Vector3 sensorOrigin;
	private Vector3 sensorOrientation;
	private Vector3 initRayDirection = new Vector3 (1, 0, 0);
	private ArrayList rays;

	void Update(){
		sensorOrigin = new Vector3 (transform.position.x, 
		                            transform.position.y, 
		                            transform.position.z);

		sensorOrientation = transform.rotation * initRayDirection;
		RaycastHit hitInfo;
		float ini_angle = 0f;
		float rot_angle = 180f / sectors;
		
		while (ini_angle <= 180) {
			Debug.DrawRay (sensorOrigin, sensorOrientation * radius);
			sensorOrientation = Quaternion.AngleAxis( -rot_angle, Vector3.up ) * sensorOrientation;
			ini_angle += rot_angle;
		}
	}
	
	private float endX(float x, float angle_deg, float distance){
		float angle_rad = Mathf.Deg2Rad * angle_deg;
		return x - distance * Mathf.Sin (angle_rad);
	}

	private float endY(float y, float angle_deg, float distance){
		float angle_rad = Mathf.Deg2Rad * angle_deg;
		return y - distance * Mathf.Cos (angle_rad);
	}	
}
