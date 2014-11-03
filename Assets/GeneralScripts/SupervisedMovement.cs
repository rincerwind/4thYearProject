using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public bool recordMovement;
	public bool debug_movement;

	private NeuralNetwork n;
	private ArrayList targetValues;
	private ArrayList initialInputs;
	private GameObject target;

	// Use this for initialization
	void Start () {
		n = gameObject.GetComponent<NeuralNetwork> ();
		target = GameObject.FindGameObjectWithTag("Goal");
		targetValues = new ArrayList ();
		initialInputs = new ArrayList ();
	}

	// Update is called once per frame
	void FixedUpdate () {
		LA.Matrix<float> inputs;
		LA.Matrix<float> outputs;
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		// Recording Phase
		if (recordMovement && !n.TrainingPhase) {
			float deltaX = target.transform.position.x - transform.position.x;
			float deltaZ = target.transform.position.z - transform.position.z;

			targetValues.Add (horizontalMovement);
			targetValues.Add (verticalMovement);
			initialInputs.Add (deltaX);
			initialInputs.Add (deltaZ);
		}

		inputs = LA.Matrix<float>.Build.Dense (1, n.numInputs, new float[]{
				target.transform.position.x - transform.position.x,
				target.transform.position.z - transform.position.z});

		// Learning Phase
		if ( !recordMovement && n.TrainingPhase ) {
			n.LearningPhase(initialInputs, targetValues, n.allowedError);
			n.TrainingPhase = false;
		}

		// Neural Net in action
		if ( !debug_movement && !recordMovement && !n.TrainingPhase ){
			outputs = n.ComputeOutputs(inputs);
			direction.x = outputs[0,0];
			direction.z = outputs[0,1];
		}

		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (direction * moveSpeed);

		//Debug.Log ("Clear");
		//if (transform.position.y < -1)
		//	player_die ();
	}
}
