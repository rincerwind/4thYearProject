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
	public void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		sm = gameObject.GetComponent<SupervisedMotion>();
		target = GameObject.FindGameObjectWithTag("Goal");
		sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();
	}

	private float vectorDiff(Vector3 targetDirection, Vector3 lookDirection){
			
			// the vector perpendicular to referenceForward (90 degrees clockwise)
			// (used to determine if angle is positive or negative)
		Vector3 referenceRight= Vector3.Cross(Vector3.up, lookDirection);
		if ( float.IsNaN(referenceRight.x) 
		    || float.IsNaN(referenceRight.y) 
		    || float.IsNaN(referenceRight.z) )
			return 0f;
			
		// Get the angle in degrees between 0 and 180
		float angle = Vector3.Angle(targetDirection, lookDirection);
		
		// Determine if the degree value should be negative. Here, a positive value
		// from the dot product means that our vector is on the right of the reference vector
		// whereas a negative value means we're on the left.
		float sign = Mathf.Sign(Vector3.Dot(targetDirection, referenceRight));
		
		float finalAngle = sign * angle;
		return finalAngle;
	}

	private void recordSensorData(ref NeuralNetwork net, ArrayList hits, float[] rotType){
		net.allInputs.AddRange(hits);

		net.allOutputs.Add(rotType[0]);
		net.allOutputs.Add(rotType[1]);
		net.allOutputs.Add(rotType[2]);
	}

	private void recordRotation(ref NeuralNetwork net, float rotation, float angleDiff){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		net.allOutputs.Add(rotation);
		net.allInputs.Add(angleDiff);
	}

	private void recordDirection(ref NeuralNetwork net, float verticalMovement,
	                             float deltaZ){
		if (sensor == null)
			sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();

		net.allOutputs.Add (verticalMovement);
		net.allInputs.Add (deltaZ);
	}

	private void trainNetworks(){
		foreach (NeuralNetwork net in nets)
			net.LearningPhase(net.allInputs, net.allOutputs, net.allowedError);
		TrainingPhase = false;
	}

	private Vector3 GetNewDirection(NeuralNetwork net, float deltaZ){
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, new float[]{
									deltaZ });
		LA.Matrix<float> outputs = net.ComputeOutputs2(inputs);
		return new Vector3(0, 0, outputs[0,0]);
	}
	
	private float[] GetNewSensorData(NeuralNetwork net, ArrayList hits){
		float[] input_array = new float[net.numInputs];
		
		for(int i = 0; i < hits.Count; i++){
			float hit = (float)hits[i];
			input_array[i] = hit; 
		}

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);
		
		LA.Matrix<float> outputs = net.ComputeOutputs2(inputs);
		float[] output = new float[]{outputs[0,0], outputs[0,1], outputs[0,2]};
		return output;
	}

	private float GetNewRotation(NeuralNetwork net, float angleDiff){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		float[] input_array = new float[net.numInputs];
		input_array[0] = angleDiff;

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);

		LA.Matrix<float> outputs = net.ComputeOutputs2(inputs);
		return outputs[0,0];
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		float rotAmount = 0f;
		float deltaZ = 0f;
		Vector3 direction = new Vector3 (0, 0, verticalMovement);
		ArrayList hits = sensor.collisionCheck();
		Vector3 goalDirection;
		float angleDiff;

		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		if( recordMovement ){
			RaycastHit hitInfo;
			if( WorldManager.currentLevel == 1 ){
				Ray r = new Ray(transform.position, transform.rotation * Vector3.forward*100);
				if(Physics.Raycast( r, out hitInfo, 100, 1<<LayerMask.NameToLayer("Goal") )){
					Debug.DrawRay(transform.position, transform.rotation * Vector3.forward*100, Color.red);
					sm.OnTriggerEnter(target.collider);
				}
				else
					Debug.DrawRay(transform.position, transform.rotation * Vector3.forward*100, Color.white);
			}
			if( WorldManager.currentLevel == 2 ){
				GameObject trainWall = GameObject.FindGameObjectWithTag("TrainWall");

				if( trainWall != null && Input.GetKeyDown(KeyCode.Space) )
					sm.OnTriggerEnter(trainWall.collider);
			}
			Debug.DrawLine(transform.position, target.transform.position);
		}

		//deltaZ = Mathf.Abs(target.transform.position.z - transform.position.z);
		deltaZ = Vector3.Distance(target.transform.position, transform.position);
		rotAmount = horizontalMovement;
		goalDirection = target.transform.position - transform.position;
		angleDiff = vectorDiff(goalDirection, transform.forward);

		// Handle direction recording
		if ( WorldManager.currentLevel == 0 && recordMovement 
		    && !TrainingPhase && (direction.z != 0f || rotAmount != 0f) )
			recordDirection(ref nets[0], verticalMovement, deltaZ );

		// Handle rotation data recording
		if ( WorldManager.currentLevel == 1 &&  recordMovement 
		    && !TrainingPhase && (direction.z != 0f || rotAmount != 0f) )
			recordRotation(ref nets[1], rotAmount, angleDiff);

		// Handle sensor data recording
		if ( WorldManager.currentLevel == 2 && recordMovement 
		    && !TrainingPhase && (direction.z > 0f || rotAmount != 0f) ){
			float[] rotType = {0f,0f,0f};

			if( (int)(rotAmount*10) < 0f )
				rotType[0] = 1f;
			else if( (int)(rotAmount*10) > 0f )
				rotType[2] = 1f;
			else
				rotType[1] = 1f;

			recordSensorData(ref nets[2], hits, rotType);
		}

		// Handle Network training
		if ( !recordMovement && TrainingPhase )
			trainNetworks();

		// Controller
		if ( !debugMovement && !recordMovement && !TrainingPhase ){
			float[] rot_prob = GetNewSensorData(nets[2], hits);
			int max_val = -1;
			int max_pos = -1;

			if( (int)(rot_prob[0]*10) > (int)(rot_prob[1]*10) ){
				max_val = -(int)(rot_prob[0]*10);
				max_pos = 0;
			}
			else {
				max_val = 0;
				max_pos = 1;
			}

			if( (int)(rot_prob[max_pos]*10) < (int)(rot_prob[2]*10) ){
				max_val = (int)(rot_prob[2]*10);
				max_pos = 2;
			}

			if( Mathf.Abs(max_val) > 5 )
				rotAmount = max_val / 10.0f;
			else
				rotAmount = GetNewRotation(nets[1], angleDiff);

			direction.z = 0.30f; // Test code
			print ( new Vector3(rot_prob[0], rot_prob[1], rot_prob[2]) );
		}

		sm.rotate(rotAmount * Time.deltaTime * sm.rotateSpeed);
		sm.move(direction);
	}
}
