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
	}

	private float[,] ListToMultiArray(ArrayList a){
		float[] temp = (float[])a [0];
		int numInputs = temp.Length;
		int numTestCases = a.Count;
		float[,] arrayCpy = new float[numTestCases, numInputs];

		for (int i = 0; i < numTestCases; i++) {
			temp = (float[])a[i];
			for (int j = 0; j < numInputs; j++)
				arrayCpy[i,j] = temp[j];
		}

		return arrayCpy;
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		// Recording Phase
		if (recordMovement && !n.TrainingPhase) {
			targetValues.Add (new float[]{horizontalMovement, verticalMovement});
			initialInputs.Add (new float[]{transform.position.x, transform.position.z, 
				target.transform.position.x, target.transform.position.z});
		}

		LA.Matrix<float> inputs;
		LA.Matrix<float> targetOutputs;
		LA.Matrix<float> outputs;

		inputs = LA.Matrix<float>.Build.Dense (1, 4, new float[]{transform.position.x, transform.position.z,
													target.transform.position.x, target.transform.position.z});

		// Learning Phase
		if ( !recordMovement && n.TrainingPhase ) {
			inputs = LA.Matrix<float>.Build.Dense(1, 4, (float[])initialInputs[0] );
			outputs = n.ComputeOutputs(inputs);

			for(int i = 1; i <= 130; i++){
				for( int j = 0; j < initialInputs.Count; j++ ){
					inputs = LA.Matrix<float>.Build.DenseOfColumnMajor(1, 4, (float[])initialInputs[j] );
					targetOutputs = LA.Matrix<float>.Build.DenseOfColumnMajor(1, 2, (float[])targetValues[j] );
					n.UpdateWeights(targetOutputs, n.eta, n.alpha);
					outputs = n.ComputeOutputs(inputs);
				}
			}
			n.TrainingPhase = false;
			targetValues = new ArrayList ();
			initialInputs = new ArrayList ();
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
		
		//if (transform.position.y < -1)
		//	player_die ();
	}
}
