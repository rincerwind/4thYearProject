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
	private ArrayList targetValues;
	private ArrayList initialInputs;
	private GameObject target;

	// Use this for initialization
	void Start () {
		debug = false;
		n = gameObject.GetComponent<NeuralNetwork> ();
		target = GameObject.FindGameObjectWithTag("Goal");
		targetValues = new ArrayList ();
		initialInputs = new ArrayList ();

		/*targetValues.Add (new float[]{0,-1});
		initialInputs.Add (new float[]{0,0,0,-11});
		
		targetValues.Add (new float[]{0,1});
		initialInputs.Add (new float[]{0,0,0,11});
		
		targetValues.Add (new float[]{-1,0});
		initialInputs.Add (new float[]{0,0,-11,0});
		
		targetValues.Add (new float[]{1,0});
		initialInputs.Add (new float[]{0,0,11,0});*/
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		// Recording Phase
		if (recordMovement && !n.TrainingPhase) {
			float deltaX = Mathf.Abs(target.transform.position.x - transform.position.x);
			float deltaZ = Mathf.Abs(target.transform.position.z - transform.position.z);

			//horizontalMovement = (deltaX <= 5)? 0 : horizontalMovement;
			//verticalMovement = (deltaZ <= 5)? 0 : verticalMovement;

			targetValues.Add (new float[]{horizontalMovement, verticalMovement});
			initialInputs.Add (new float[]{deltaX, deltaZ});
		}

		LA.Matrix<float> inputs;
		LA.Matrix<float> targetOutputs;
		LA.Matrix<float> outputs;

		inputs = LA.Matrix<float>.Build.Dense (1, n.numInputs, new float[]{
			Mathf.Abs(target.transform.position.x - transform.position.x),
			Mathf.Abs(target.transform.position.z - transform.position.z)});

		// Learning Phase
		if ( !recordMovement && n.TrainingPhase ) {
			int i;
			int count = initialInputs.Count;
			for( i = 1; i <= n.numIterations && count > 0; i++){
				count = 0;
				for( int j = 0; j < initialInputs.Count; j++ ){
					inputs = LA.Matrix<float>.Build.DenseOfColumnMajor(1, n.numInputs, (float[])initialInputs[j] );
					targetOutputs = LA.Matrix<float>.Build.DenseOfColumnMajor(1, n.numOutputs, (float[])targetValues[j] );
					outputs = n.ComputeOutputs(inputs);

					if( n.error(targetOutputs) <= 0.01 ){
						continue;
					}
					count++;
					// count Mean Squared Error here and try to minimize it
					n.UpdateWeights(targetOutputs, n.eta, n.alpha);
					outputs = n.ComputeOutputs(inputs);
				}
			}
			Debug.Log(i);
			n.TrainingPhase = false;
			//targetValues = new ArrayList ();
			//initialInputs = new ArrayList ();
			return;
		}

		// Neural Net in action
		if ( !recordMovement && !n.TrainingPhase )
		{
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
