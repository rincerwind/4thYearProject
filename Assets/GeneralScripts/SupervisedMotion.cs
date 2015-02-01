using UnityEngine;
using System.Collections;

public class SupervisedMotion : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public float rotateSpeed;

	private GameObject target;
	private NextGoal g;

	public void setNextGoal( NextGoal new_g ) {
		g = new_g;
	}
	public NextGoal getNextGoal(){ return g; }

	public void move(Vector3 direction){
		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (transform.rotation * direction * moveSpeed);
	}

	public void rotate(float amount){
		rigidbody.AddTorque(transform.up * amount, ForceMode.Acceleration);
	}

	// Use this for initialization
	public void Start () {
		target = GameObject.FindGameObjectWithTag("Goal");
		g = target.GetComponent<NextGoal>();
	}

	// On level 0, teach the Navigation Net
	// On level 1, teach the Collision Avoidance Net
	// On level 2, test going towards a goal
	// On level 3, test going towards a goal and avoiding small objects
	// On level 4, test going towards a goal and avoiding large objects
	// On level 5, test going towards a goal and avoiding a mixed-size objects
	public void OnTriggerEnter(Collider c){
		switch (WorldManager.currentLevel){
			case 0: case 1: case 2:
				if( c.transform.tag == "Goal" ){
					if( g == null || ( g != null && g.isLastGoal() ) )
						WorldManager.CompleteLevel();
					else{
						g.goToNextGoal();
						target.transform.position = (g.getCurrentGoal()).position;
					}
				}
				break;
			default:
				WorldManager.CompleteLevel();
				break;
		}
	}
}// end of class
