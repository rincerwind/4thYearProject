﻿using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public float rotateSpeed;
	public bool recordMovement;
	public bool debugMovement;
	
	private NeuralNetwork[] nets;
	private ArrayList targetValues;
	private ArrayList initialInputs;
	private GameObject target;
	private NextGoal g;

	// Use this for initialization
	void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		target = GameObject.FindGameObjectWithTag("Goal");
		targetValues = new ArrayList ();
		initialInputs = new ArrayList ();
		g = target.GetComponent<NextGoal>();
	}

	// Update is called once per frame
	void FixedUpdate () {
		LA.Matrix<float> inputs;
		LA.Matrix<float> outputs;
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		// Recording Phase
		if (recordMovement && !nets[0].TrainingPhase) {
			float deltaX = target.transform.position.x - transform.position.x;
			float deltaZ = target.transform.position.z - transform.position.z;

			targetValues.Add (horizontalMovement);
			targetValues.Add (verticalMovement);
			initialInputs.Add (deltaX);
			initialInputs.Add (deltaZ);
		}

		// Learning Phase of Navigation Net
		if ( !recordMovement && nets[0].TrainingPhase ) {
			nets[0].LearningPhase(initialInputs, targetValues, nets[0].allowedError);
			nets[0].TrainingPhase = false;
		}

		inputs = LA.Matrix<float>.Build.Dense (1, 
		    nets[0].numInputs, new float[]{
			target.transform.position.x - transform.position.x,
			target.transform.position.z - transform.position.z});

		// Navigation Net in action
		if ( !debugMovement && !recordMovement && !nets[0].TrainingPhase ){
			outputs = nets[0].ComputeOutputs(inputs);
			direction.x = outputs[0,0];
			direction.z = outputs[0,1];
		}

		// Manual rotation
		float amount = 0f;
		if ( Input.GetKey("c") )
			amount = -1 * Time.deltaTime * rotateSpeed;

		if ( Input.GetKey("v") )
			amount = 1 * Time.deltaTime * rotateSpeed;

		rigidbody.AddTorque(transform.up * amount, ForceMode.Acceleration);

		// Manual movement
		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (transform.rotation * direction * moveSpeed);
	}// end of FixedUpdate

	// On level 0, teach the Navigation Net
	// On level 1, teach the Collision Avoidance Net
	// On level 2, test going towards a goal
	// On level 3, test going towards a goal and avoiding small objects
	// On level 4, test going towards a goal and avoiding large objects
	// On level 5, test going towards a goal and avoiding a mixed-size objects
	void OnTriggerEnter(Collider c){
		switch (WorldManager.currentLevel){
			case 0:
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
