using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class NetworkController : MonoBehaviour {

	public bool recordMovement;
	public bool debugMovement;
	public bool TrainingPhase;

	private NeuralNetwork[] nets;
	private GameObject target;
	private SupervisedMotion sm;

	// Use this for initialization
	void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		sm = gameObject.GetComponent<SupervisedMotion>();
		target = GameObject.FindGameObjectWithTag("Goal");
	}

	private void recordDirection(ref NeuralNetwork net, float horizontalMovement, float verticalMovement){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		float deltaX = target.transform.position.x - transform.position.x;
		float deltaZ = target.transform.position.z - transform.position.z;
			
		net.allOutputs.Add(horizontalMovement);
		net.allOutputs.Add (verticalMovement);
		net.allInputs.Add (deltaX);
		net.allInputs.Add (deltaZ);
	}

	private void trainNetworks(){
		//foreach (NeuralNetwork net in nets)
			nets[0].LearningPhase(nets[0].allInputs, nets[0].allOutputs, nets[0].allowedError);
		TrainingPhase = false;
	}

	private Vector3 GetNewDirection(NeuralNetwork net){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, new float[]{
									target.transform.position.x - transform.position.x,
									target.transform.position.z - transform.position.z});
		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return new Vector3(outputs[0,0], 0, outputs[0,1]);
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);
		float amount = 0f;

		// Handle manual rotation
		if ( Input.GetKey("c") )
			amount = -1 * Time.deltaTime * sm.rotateSpeed;
		
		if ( Input.GetKey("v") )
			amount = 1 * Time.deltaTime * sm.rotateSpeed;

		// Handle direction recording
		if (recordMovement && !TrainingPhase)
			recordDirection(ref nets[0], horizontalMovement, verticalMovement);

		// Handle Network training
		if ( !recordMovement && TrainingPhase )
			trainNetworks();

		// Obtain new direction
		if ( !debugMovement && !recordMovement && !TrainingPhase )
			direction = GetNewDirection(nets[0]);

		sm.rotate(amount);
		sm.move(direction);
	}
}
