using UnityEngine;
using System.Collections;

public class FollowPlayer : MonoBehaviour {

	public Vector3 cameraFollowOffset = new Vector3(0,0,0);
	public Camera followCamera;
	
	void Start(){
		followCamera = Camera.main;
	}
	
	void Update(){
		if (followCamera == null)
			followCamera = Camera.main;
		followCamera.transform.position = transform.position + cameraFollowOffset;
	}
}

