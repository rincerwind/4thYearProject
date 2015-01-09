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
	private RadialGridSensor sensor;

	// Use this for initialization
	void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		sm = gameObject.GetComponent<SupervisedMotion>();
		target = GameObject.FindGameObjectWithTag("Goal");
		sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();
	}

	private void recordSensorData(ref NeuralNetwork net, ArrayList hits,
	                              float horizontalMovement, float verticalMovement, 
	                              Vector3 curr_rotation){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		float deltaX = target.transform.position.x - transform.position.x;
		float deltaZ = target.transform.position.z - transform.position.z;
		
		net.allOutputs.Add(horizontalMovement);
		net.allOutputs.Add (verticalMovement);

		net.allInputs.Add (deltaX);
		net.allInputs.Add (deltaZ);
		net.allInputs.Add (curr_rotation.x);
		net.allInputs.Add (curr_rotation.y);
		net.allInputs.Add (curr_rotation.z);
		net.allInputs.AddRange(hits);
	}

	private void recordDirection(ref NeuralNetwork net, float horizontalMovement, float verticalMovement){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		if (sensor == null)
			sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();

		float deltaX = target.transform.position.x - transform.position.x;
		float deltaZ = target.transform.position.z - transform.position.z;
			
		net.allOutputs.Add(horizontalMovement);
		net.allOutputs.Add (verticalMovement);
		net.allInputs.Add (deltaX);
		net.allInputs.Add (deltaZ);
	}

	private void trainNetworks(){
		foreach (NeuralNetwork net in nets)
			net.LearningPhase(net.allInputs, net.allOutputs, net.allowedError);
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

	private float GetNewRotation(NeuralNetwork net, ArrayList hits, Vector3 rotation){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		float[] input_array = new float[net.numInputs];
		input_array[0] = target.transform.position.x - transform.position.x;
		input_array[1] = target.transform.position.z - transform.position.z;
		input_array[2] = rotation.x;
		input_array[3] = rotation.y;
		input_array[4] = rotation.z;

		for(int i = 0; i < hits.Count; i++){
			int hit = (int)hits[0];
			input_array[5 + i] = hit; 
		}

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);

		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return outputs[0,0];
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		float amount = 0f;
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);
		Vector3 rotation = transform.rotation.eulerAngles;
		ArrayList hits = sensor.collisionCheck();

		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		if( recordMovement )
			Debug.DrawLine(transform.position, target.transform.position);

		// Handle manual rotation
		if ( Input.GetKey("c") )
			amount = -1 * Time.deltaTime * sm.rotateSpeed;
		
		if ( Input.GetKey("v") )
			amount = 1 * Time.deltaTime * sm.rotateSpeed;

		// Handle direction recording
		if ( recordMovement && !TrainingPhase && (direction.x != 0f || direction.z != 0f) )
			recordDirection(ref nets[0], horizontalMovement, verticalMovement);

		// Handle sensor data recording
		if ( recordMovement && !TrainingPhase && (direction.x != 0f || direction.z != 0f || amount != 0f) )
			recordSensorData(ref nets[1], hits,
			                 horizontalMovement, verticalMovement, 
			                 rotation);

		// Handle Network training
		if ( !recordMovement && TrainingPhase )
			trainNetworks();

		// Obtain new direction
		if ( !debugMovement && !recordMovement && !TrainingPhase ){
			direction = GetNewDirection(nets[0]);
			amount = GetNewRotation(nets[1], hits, rotation);
		}

		sm.rotate(amount);
		sm.move(direction);
	}
}
