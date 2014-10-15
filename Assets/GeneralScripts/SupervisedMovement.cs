using UnityEngine;
using System.Collections;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public bool recordMovement;

	private Vector3 direction;
	private bool debug;

	// Use this for initialization
	void Start () {
		debug = true;
		recordMovement = false;
	}
	
	// Update is called once per frame
	void FixedUpdate () {
		direction = new Vector3 (Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical"));

		
		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (direction * moveSpeed);
		
		//if (transform.position.y < -1)
		//	player_die ();
		
		if( debug == true )
			Debug.Log (direction);
	}
}
