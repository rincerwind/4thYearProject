using UnityEngine;
using System.Collections;

public class CameraZoom : MonoBehaviour {

	public float zoomStep = 20;
	public float maxZoom = 40;
	public float minZoom = 5;
	
	void Update () 
	{
		float scroll = Input.GetAxis("Mouse ScrollWheel");
		if ((scroll != 0.0f) && (camera.orthographicSize - scroll * zoomStep >= maxZoom))
			camera.orthographicSize = maxZoom;
		else if ((scroll != 0.0f) && (camera.orthographicSize - scroll * zoomStep <= minZoom))
			camera.orthographicSize = minZoom;
		else if(scroll != 0.0f)
			camera.orthographicSize -= scroll*zoomStep;
	}
}
