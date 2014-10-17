using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public bool recordMovement;
	
	private bool debug;
	private NeuralNetwork n;
	private IList targetValues;

	// Use this for initialization
	void Start () {
		debug = false;
		n = gameObject.GetComponent<NeuralNetwork> ();
		targetValues = new ArrayList ();
	}
	
	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		if (recordMovement && !n.TrainingPhase) {
			targetValues.Add (new float[]{horizontalMovement, verticalMovement});
			Debug.Log(direction.z);
		}

		LA.Matrix<float> inputs;
		LA.Matrix<float> targetOutputs;
		LA.Matrix<float> outputs;

		Vector3 currentPosition = transform.position;
		inputs = LA.Matrix<float>.Build.Dense (1, 2, new float[]{currentPosition.x, currentPosition.z});

		if ( !recordMovement && n.TrainingPhase ) {
			outputs = n.ComputeOutputs (inputs);
			int count = 0;
			while( (0.0 != outputs[0,0] + 0.1 || -1.0 != outputs[0,1] + 0.1) && count <= 3500 ) {
				targetOutputs = LA.Matrix<float>.Build.Dense (1, 2, new float[]{0.0f, -1.0f});
				n.UpdateWeights (targetOutputs, 0.9f, 1.0f);	
				outputs = n.ComputeOutputs (inputs);
				count++;
			}
			Debug.Log(count);
			n.TrainingPhase = false;
		}

		if ( !recordMovement && !n.TrainingPhase )
		{
			outputs = n.ComputeOutputs(inputs);
			direction.x = outputs[0,0];
			direction.z = outputs[0,1];
		}

		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (direction * moveSpeed);
		
		//if (transform.position.y < -1)
		//	player_die ();
	}
}
