# Design Document: AgriEdge-Link

## Overview

AgriEdge-Link is a comprehensive agricultural technology platform consisting of an Android mobile application and a cloud-based backend system. The platform enables smallholder farmers to diagnose crop diseases offline using edge AI, interact through voice in regional languages, and connect to agricultural markets through the Beckn/ONDC protocol.

The system is designed with an offline-first architecture, ensuring core diagnostic functionality works without internet connectivity while seamlessly syncing data and enabling market features when connectivity is available.

### Key Design Principles

1. **Offline-First**: Core diagnostic features must work without internet connectivity
2. **Voice-First**: Primary interaction mode is voice in regional languages
3. **Edge AI**: Machine learning inference happens entirely on-device
4. **Minimal Resource Usage**: Optimized for entry-level Android devices (3GB RAM, Snapdragon 665)
5. **Secure by Default**: All data encrypted at rest and in transit
6. **Scalable Backend**: Cloud services designed to handle millions of farmers
7. **Open Standards**: Integration with Beckn/ONDC for market connectivity

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Android Application                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ UI Layer   │  │ Business     │  │ Data Layer       │   │
│  │ (Compose)  │◄─┤ Logic Layer  │◄─┤ (Room + Files)   │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│         │              │                     │              │
│         │              ▼                     │              │
│         │      ┌──────────────┐             │              │
│         │      │  TFLite ML   │             │              │
│         │      │    Engine    │             │              │
│         │      └──────────────┘             │              │
└─────────┴──────────────┬───────────────────┴──────────────┘
                         │
                         │ HTTPS/TLS 1.3
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Backend Services                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ API Gateway  │  │ Auth Service │  │ Sync Service     │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │                  │                    │           │
│  ┌──────┴──────────────────┴────────────────────┴────────┐ │
│  │           Core Application Services                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │  │ User Profile │  │  Telemetry   │  │   Market    │ │ │
│  │  │   Service    │  │   Service    │  │  Connector  │ │ │
│  │  └──────────────┘  └──────────────┘  └──────┬──────┘ │ │
│  └───────────────────────────────────────────────┼────────┘ │
│                                                   │          │
│  ┌────────────────────────────────────────────────┼────────┐│
│  │              Data Layer                        │        ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────▼──────┐││
│  │  │  PostgreSQL  │  │  Cloud       │  │  Beckn/ONDC   │││
│  │  │   Database   │  │  Storage     │  │   Network     │││
│  │  └──────────────┘  └──────────────┘  └───────────────┘││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
                         │
                         │ External APIs
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼─────┐                  ┌─────▼──────┐
    │ Bhashini │                  │   SMS      │
    │   API    │                  │  Gateway   │
    └──────────┘                  └────────────┘
```


### Integration Points

1. **Bhashini API**: Government-provided multilingual AI services for speech-to-text and text-to-speech
2. **Beckn/ONDC Network**: Decentralized commerce protocol for market connectivity
3. **Cloud Storage**: AWS S3 or Google Cloud Storage for diagnostic images and model updates
4. **SMS Gateway**: For OTP delivery during authentication
5. **Firebase Cloud Messaging**: For push notifications about transactions and reminders

## Android Application Design

### Technology Stack

#### Core Technologies
- **Language**: Kotlin 1.9+
- **Minimum SDK**: Android 8.0 (API 26)
- **Target SDK**: Android 14 (API 34)
- **Build System**: Gradle with Kotlin DSL

#### UI Framework
- **Jetpack Compose**: Modern declarative UI framework
- **Material Design 3**: UI components and theming
- **Compose Navigation**: Type-safe navigation
- **Accompanist**: Compose utilities (permissions, system UI controller)

#### Architecture Components
- **ViewModel**: UI state management
- **LiveData/StateFlow**: Reactive data streams
- **Lifecycle**: Lifecycle-aware components
- **WorkManager**: Background task scheduling
- **DataStore**: Key-value storage for preferences

#### Local Data Persistence
- **Room Database**: SQLite abstraction for structured data
- **Encrypted SharedPreferences**: Secure key-value storage
- **File-based Storage**: For images and ML models

#### Machine Learning
- **TensorFlow Lite**: On-device ML inference
- **TFLite Support Library**: Image preprocessing utilities
- **TFLite GPU Delegate**: Hardware acceleration (when available)
- **TFLite Model Metadata**: Model introspection

#### Networking
- **Retrofit**: Type-safe HTTP client
- **OkHttp**: HTTP client with interceptors
- **Moshi**: JSON serialization
- **Coil**: Image loading and caching

#### Dependency Injection
- **Hilt**: Compile-time dependency injection

#### Camera
- **CameraX**: Camera API abstraction
- **ML Kit Vision**: Image analysis utilities

#### Testing
- **JUnit 5**: Unit testing framework
- **Kotest**: Property-based testing
- **MockK**: Mocking framework
- **Espresso**: UI testing
- **Robolectric**: Android unit tests without emulator

### Architecture Pattern

The application follows **Clean Architecture** with **MVVM** presentation pattern:

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Compose    │  │  ViewModel   │  │  UI State    │ │
│  │   Screens    │◄─┤   (State)    │◄─┤   Models     │ │
│  └──────────────┘  └──────┬───────┘  └──────────────┘ │
└────────────────────────────┼──────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────┐
│                    Domain Layer                        │
│  ┌──────────────┐  ┌──────▼───────┐  ┌──────────────┐│
│  │   Domain     │  │   Use Cases  │  │  Repository  ││
│  │   Models     │◄─┤  (Business   │◄─┤  Interfaces  ││
│  │              │  │    Logic)    │  │              ││
│  └──────────────┘  └──────────────┘  └──────────────┘│
└────────────────────────────┬──────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────┐
│                     Data Layer                         │
│  ┌──────────────┐  ┌──────▼───────┐  ┌──────────────┐│
│  │  Repository  │  │  Data        │  │   Remote     ││
│  │  Impl        │◄─┤  Sources     │◄─┤   API        ││
│  │              │  │  (Room/File) │  │              ││
│  └──────────────┘  └──────────────┘  └──────────────┘│
└──────────────────────────────────────────────────────┘
```

#### Layer Responsibilities

**Presentation Layer**:
- Compose UI screens and components
- ViewModels managing UI state
- UI event handling
- Navigation logic

**Domain Layer**:
- Business logic encapsulated in use cases
- Domain models (pure Kotlin classes)
- Repository interfaces (dependency inversion)
- Business rules and validation

**Data Layer**:
- Repository implementations
- Data source abstractions (local/remote)
- Data models and mappers
- Caching strategies

### Module Structure

```
app/
├── src/
│   ├── main/
│   │   ├── java/com/agriedge/
│   │   │   ├── di/                    # Dependency injection modules
│   │   │   ├── presentation/
│   │   │   │   ├── diagnosis/         # Diagnosis feature
│   │   │   │   │   ├── camera/
│   │   │   │   │   ├── result/
│   │   │   │   │   └── history/
│   │   │   │   ├── market/            # Market feature
│   │   │   │   │   ├── search/
│   │   │   │   │   ├── buyers/
│   │   │   │   │   ├── coldstorage/
│   │   │   │   │   └── equipment/
│   │   │   │   ├── auth/              # Authentication
│   │   │   │   ├── profile/           # User profile
│   │   │   │   ├── voice/             # Voice interface
│   │   │   │   └── common/            # Shared UI components
│   │   │   ├── domain/
│   │   │   │   ├── model/             # Domain models
│   │   │   │   ├── repository/        # Repository interfaces
│   │   │   │   └── usecase/           # Use cases
│   │   │   │       ├── diagnosis/
│   │   │   │       ├── market/
│   │   │   │       ├── auth/
│   │   │   │       └── sync/
│   │   │   └── data/
│   │   │       ├── local/
│   │   │       │   ├── database/      # Room database
│   │   │       │   ├── datastore/     # Preferences
│   │   │       │   └── file/          # File storage
│   │   │       ├── remote/
│   │   │       │   ├── api/           # API interfaces
│   │   │       │   ├── dto/           # Data transfer objects
│   │   │       │   └── beckn/         # Beckn protocol client
│   │   │       ├── ml/
│   │   │       │   ├── classifier/    # TFLite classifier
│   │   │       │   └── preprocessor/  # Image preprocessing
│   │   │       └── repository/        # Repository implementations
│   │   └── assets/
│   │       ├── models/                # TFLite models
│   │       └── i18n/                  # Localization files
│   └── test/                          # Unit tests
│   └── androidTest/                   # Instrumentation tests
```


### Offline-First Data Architecture

#### Data Flow Strategy

```
User Action
    │
    ▼
┌─────────────────┐
│   ViewModel     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Use Case     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Repository    │─────►│ Local Cache  │ (Primary)
└────────┬────────┘      └──────────────┘
         │
         │ (Background sync when online)
         │
         ▼
┌─────────────────┐
│  Remote API     │ (Secondary)
└─────────────────┘
```

#### Sync Strategy

**Write-Through Cache Pattern**:
1. All writes go to local database first (immediate consistency)
2. Writes are queued for background sync
3. Sync happens opportunistically when network available
4. Conflict resolution uses "last write wins" with server timestamp

**Read Strategy**:
1. Always read from local database (fast, offline-capable)
2. Background refresh from server when online
3. Update local cache with server data
4. Notify UI of updates via Flow/LiveData

#### Sync Queue Implementation

```kotlin
@Entity(tableName = "sync_queue")
data class SyncQueueItem(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val entityType: EntityType,  // DIAGNOSIS, PROFILE, RATING, etc.
    val entityId: String,
    val operation: Operation,     // CREATE, UPDATE, DELETE
    val payload: String,          // JSON serialized entity
    val timestamp: Long,
    val retryCount: Int = 0,
    val status: SyncStatus        // PENDING, IN_PROGRESS, FAILED, COMPLETED
)

enum class EntityType {
    DIAGNOSIS, USER_PROFILE, TRANSACTION, RATING, TELEMETRY
}

enum class Operation {
    CREATE, UPDATE, DELETE
}

enum class SyncStatus {
    PENDING, IN_PROGRESS, FAILED, COMPLETED
}
```

**Sync Worker** (using WorkManager):
```kotlin
class SyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        // 1. Check network connectivity
        // 2. Fetch pending items from sync queue
        // 3. Process items in order (FIFO)
        // 4. Retry failed items with exponential backoff
        // 5. Update sync status
        // 6. Remove completed items
        
        return Result.success()
    }
}
```

**Constraints**:
- Sync only on WiFi or metered network (user preference)
- Require battery not low
- Exponential backoff: 30s, 2m, 10m for retries
- Maximum 3 retry attempts before marking as failed

### ML Model Integration

#### Model Architecture

**Base Model**: MobileNetV3-Small with custom classification head
- **Input**: 224x224 RGB images
- **Output**: 40+ disease classes + confidence scores
- **Model Size**: ~45MB (quantized)
- **Inference Time**: <3 seconds on Snapdragon 665

**Model Quantization**:
- Post-training quantization (INT8)
- Dynamic range quantization for weights
- Maintains >85% accuracy after quantization

#### TFLite Integration

```kotlin
class DiseaseClassifier(
    private val context: Context,
    private val modelPath: String = "models/crop_disease_classifier.tflite"
) {
    private var interpreter: Interpreter? = null
    private val inputShape = intArrayOf(1, 224, 224, 3)
    private val outputShape = intArrayOf(1, 40)
    
    fun initialize() {
        val options = Interpreter.Options().apply {
            // Use GPU delegate if available
            if (isGpuAvailable()) {
                addDelegate(GpuDelegate())
            }
            // Use NNAPI delegate as fallback
            useNNAPI = true
            numThreads = 4
        }
        
        val model = loadModelFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }
    
    fun classify(bitmap: Bitmap): ClassificationResult {
        val preprocessed = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(40) }
        
        interpreter?.run(preprocessed, output)
        
        return parseOutput(output[0])
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // 1. Resize to 224x224
        // 2. Normalize pixel values [0, 255] -> [0, 1]
        // 3. Convert to ByteBuffer
        // 4. Apply model-specific preprocessing
    }
    
    fun close() {
        interpreter?.close()
    }
}

data class ClassificationResult(
    val topPredictions: List<Prediction>,
    val inferenceTime: Long
)

data class Prediction(
    val diseaseId: String,
    val diseaseName: String,
    val confidence: Float,
    val cropType: CropType
)
```

#### Model Update Strategy

**Over-The-Air (OTA) Updates**:
1. Backend publishes new model version with metadata
2. App checks for updates on WiFi connection
3. Downloads model to cache directory
4. Validates model integrity (checksum)
5. Swaps model atomically
6. Falls back to previous model if validation fails

**Model Versioning**:
```kotlin
data class ModelMetadata(
    val version: String,
    val checksum: String,
    val size: Long,
    val minAppVersion: String,
    val cropTypes: List<CropType>,
    val diseaseCount: Int,
    val accuracy: Float,
    val downloadUrl: String
)
```

### Camera and Image Processing Pipeline

#### CameraX Implementation

```kotlin
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    private var imageCapture: ImageCapture? = null
    private var imageAnalyzer: ImageAnalysis? = null
    
    fun startCamera(
        previewView: PreviewView,
        onImageCaptured: (Bitmap) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Preview use case
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            // Image capture use case
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setTargetRotation(previewView.display.rotation)
                .build()
            
            // Image analysis for real-time guidance
            imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        ContextCompat.getMainExecutor(context),
                        LeafDetectionAnalyzer()
                    )
                }
            
            // Bind to lifecycle
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageCapture,
                imageAnalyzer
            )
        }, ContextCompat.getMainExecutor(context))
    }
    
    fun captureImage(onImageCaptured: (Bitmap) -> Unit) {
        imageCapture?.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmap()
                    onImageCaptured(bitmap)
                    image.close()
                }
                
                override fun onError(exception: ImageCaptureException) {
                    // Handle error
                }
            }
        )
    }
}
```

#### Image Quality Validation

```kotlin
class ImageQualityValidator {
    
    fun validate(bitmap: Bitmap): ValidationResult {
        val checks = listOf(
            checkBrightness(bitmap),
            checkBlur(bitmap),
            checkResolution(bitmap),
            checkLeafPresence(bitmap)
        )
        
        val failedChecks = checks.filter { !it.passed }
        
        return if (failedChecks.isEmpty()) {
            ValidationResult.Valid
        } else {
            ValidationResult.Invalid(failedChecks.map { it.message })
        }
    }
    
    private fun checkBrightness(bitmap: Bitmap): QualityCheck {
        // Calculate average brightness
        // Reject if too dark (<30) or too bright (>220)
    }
    
    private fun checkBlur(bitmap: Bitmap): QualityCheck {
        // Use Laplacian variance to detect blur
        // Reject if variance below threshold
    }
    
    private fun checkResolution(bitmap: Bitmap): QualityCheck {
        // Ensure minimum resolution (512x512)
    }
    
    private fun checkLeafPresence(bitmap: Bitmap): QualityCheck {
        // Use simple color-based detection for green regions
        // Ensure at least 30% of image is green
    }
}

sealed class ValidationResult {
    object Valid : ValidationResult()
    data class Invalid(val reasons: List<String>) : ValidationResult()
}
```

#### Real-Time Guidance Overlay

```kotlin
class LeafDetectionAnalyzer : ImageAnalysis.Analyzer {
    
    override fun analyze(image: ImageProxy) {
        // 1. Convert to bitmap
        // 2. Detect green regions (leaf candidates)
        // 3. Calculate bounding box
        // 4. Provide guidance:
        //    - "Move closer" if leaf too small
        //    - "Center the leaf" if off-center
        //    - "Improve lighting" if too dark
        //    - "Ready to capture" if optimal
        
        image.close()
    }
}
```


### Voice Interface Integration

#### Voice Interface Integration (AWS Transcribe/Polly + Bhashini Fallback)

**Primary: Amazon Transcribe and Polly**

```kotlin
class AWSVoiceService(
    private val transcribeClient: TranscribeClient,
    private val pollyClient: PollyClient,
    private val s3Client: S3Client
) {
    
    suspend fun speechToText(
        audioData: ByteArray,
        languageCode: String  // hi-IN, mr-IN, ta-IN, te-IN, kn-IN, bn-IN
    ): SpeechToTextResponse {
        // Upload audio to S3
        val audioKey = "audio/${UUID.randomUUID()}.wav"
        s3Client.putObject {
            bucket = "agriedge-audio"
            key = audioKey
            body = ByteStream.fromBytes(audioData)
        }
        
        // Start transcription job
        val jobName = "transcribe-${UUID.randomUUID()}"
        transcribeClient.startTranscriptionJob {
            transcriptionJobName = jobName
            languageCode = languageCode
            mediaFormat = MediaFormat.Wav
            media {
                mediaFileUri = "s3://agriedge-audio/$audioKey"
            }
        }
        
        // Poll for completion
        val result = waitForTranscriptionComplete(jobName)
        
        return SpeechToTextResponse(
            transcript = result.transcript,
            confidence = result.confidence
        )
    }
    
    suspend fun textToSpeech(
        text: String,
        languageCode: String,
        voiceId: String = getVoiceForLanguage(languageCode)
    ): TextToSpeechResponse {
        val response = pollyClient.synthesizeSpeech {
            this.text = text
            this.voiceId = VoiceId.fromValue(voiceId)
            this.outputFormat = OutputFormat.Mp3
            this.languageCode = languageCode
        }
        
        return TextToSpeechResponse(
            audioContent = response.audioStream?.readBytes()?.encodeBase64() ?: "",
            audioFormat = "mp3"
        )
    }
    
    private fun getVoiceForLanguage(languageCode: String): String {
        return when (languageCode) {
            "hi-IN" -> "Aditi"  // Hindi
            "mr-IN" -> "Aditi"  // Marathi (uses Hindi voice)
            "ta-IN" -> "Aditi"  // Tamil (neural voice)
            "te-IN" -> "Aditi"  // Telugu (neural voice)
            "kn-IN" -> "Aditi"  // Kannada (neural voice)
            "bn-IN" -> "Aditi"  // Bengali (neural voice)
            else -> "Aditi"
        }
    }
}

data class SpeechToTextResponse(
    val transcript: String,
    val confidence: Float
)

data class TextToSpeechResponse(
    val audioContent: String,  // Base64 encoded audio
    val audioFormat: String
)
```

**Fallback: Bhashini API**

```kotlin
interface BhashiniService {
    
    @POST("v1/speech-to-text")
    suspend fun speechToText(
        @Body request: BhashiniSpeechToTextRequest
    ): BhashiniSpeechToTextResponse
    
    @POST("v1/text-to-speech")
    suspend fun textToSpeech(
        @Body request: BhashiniTextToSpeechRequest
    ): BhashiniTextToSpeechResponse
}

// Bhashini request/response models
data class BhashiniSpeechToTextRequest(
    val audioContent: String,
    val languageCode: String,
    val encoding: AudioEncoding,
    val sampleRateHertz: Int
)

data class BhashiniSpeechToTextResponse(
    val transcript: String,
    val confidence: Float,
    val alternatives: List<Alternative>
)
```

**Voice Service with Fallback**

```kotlin
class VoiceInterfaceManager(
    private val awsVoiceService: AWSVoiceService,
    private val bhashiniService: BhashiniService
) {
    
    suspend fun speechToText(
        audioData: ByteArray,
        languageCode: String
    ): SpeechToTextResponse {
        return try {
            // Try AWS Transcribe first
            awsVoiceService.speechToText(audioData, languageCode)
        } catch (e: Exception) {
            // Fallback to Bhashini
            Log.w("VoiceInterface", "AWS Transcribe failed, using Bhashini fallback", e)
            val bhashiniResponse = bhashiniService.speechToText(
                BhashiniSpeechToTextRequest(
                    audioContent = audioData.encodeBase64(),
                    languageCode = languageCode,
                    encoding = AudioEncoding.LINEAR16,
                    sampleRateHertz = 16000
                )
            )
            SpeechToTextResponse(
                transcript = bhashiniResponse.transcript,
                confidence = bhashiniResponse.confidence
            )
        }
    }
    
    suspend fun textToSpeech(
        text: String,
        languageCode: String
    ): TextToSpeechResponse {
        return try {
            // Try AWS Polly first
            awsVoiceService.textToSpeech(text, languageCode)
        } catch (e: Exception) {
            // Fallback to Bhashini
            Log.w("VoiceInterface", "AWS Polly failed, using Bhashini fallback", e)
            val bhashiniResponse = bhashiniService.textToSpeech(
                BhashiniTextToSpeechRequest(
                    text = text,
                    languageCode = languageCode,
                    voiceGender = VoiceGender.FEMALE,
                    speakingRate = 1.0f
                )
            )
            TextToSpeechResponse(
                audioContent = bhashiniResponse.audioContent,
                audioFormat = "mp3"
            )
        }
    }
}
```

### Amazon Bedrock Integration (Treatment Recommendations)

**Using Claude 3 or Titan for Treatment Generation**

```python
# Backend Lambda function
import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def generate_treatment_recommendation(disease_id: str, crop_type: str, language: str) -> dict:
    """
    Generate treatment recommendations using Amazon Bedrock (Claude 3)
    """
    
    # Construct prompt with agricultural knowledge
    prompt = f"""You are an expert agricultural advisor. Provide treatment recommendations for the following:

Disease: {disease_id}
Crop: {crop_type}
Language: {language}

Please provide:
1. Organic treatment options with specific products and application methods
2. Chemical treatment options with dosage and timing
3. Preventive measures for future occurrences
4. Expected recovery timeline

Format the response in {language} language, using locally available product names in India.
"""
    
    # Call Bedrock with Claude 3
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    treatment_text = response_body['content'][0]['text']
    
    # Parse and structure the response
    return {
        "disease_id": disease_id,
        "crop_type": crop_type,
        "language": language,
        "treatment_text": treatment_text,
        "generated_at": datetime.now().isoformat()
    }

def lambda_handler(event, context):
    """
    Lambda handler for treatment recommendation generation
    """
    body = json.loads(event['body'])
    
    disease_id = body['disease_id']
    crop_type = body['crop_type']
    language = body.get('language', 'en-IN')
    
    try:
        recommendation = generate_treatment_recommendation(disease_id, crop_type, language)
        
        # Cache in DynamoDB for future requests
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('TreatmentRecommendations')
        table.put_item(Item=recommendation)
        
        return {
            'statusCode': 200,
            'body': json.dumps(recommendation)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Knowledge Base Integration**

```python
def create_bedrock_knowledge_base():
    """
    Create Amazon Bedrock Knowledge Base with agricultural documents
    """
    bedrock_agent = boto3.client('bedrock-agent')
    
    # Create knowledge base
    response = bedrock_agent.create_knowledge_base(
        name='AgriEdge-Agricultural-Knowledge',
        description='Comprehensive agricultural knowledge base for crop diseases and treatments',
        roleArn='arn:aws:iam::ACCOUNT_ID:role/BedrockKnowledgeBaseRole',
        knowledgeBaseConfiguration={
            'type': 'VECTOR',
            'vectorKnowledgeBaseConfiguration': {
                'embeddingModelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1'
            }
        },
        storageConfiguration={
            'type': 'OPENSEARCH_SERVERLESS',
            'opensearchServerlessConfiguration': {
                'collectionArn': 'arn:aws:aoss:us-east-1:ACCOUNT_ID:collection/agriedge-kb',
                'vectorIndexName': 'agriedge-vector-index',
                'fieldMapping': {
                    'vectorField': 'embedding',
                    'textField': 'text',
                    'metadataField': 'metadata'
                }
            }
        }
    )
    
    return response['knowledgeBase']['knowledgeBaseId']

def query_knowledge_base(query: str, knowledge_base_id: str) -> str:
    """
    Query the knowledge base for relevant agricultural information
    """
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
    
    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={
            'text': query
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 5
            }
        }
    )
    
    # Combine retrieved documents
    context = "\n\n".join([
        result['content']['text'] 
        for result in response['retrievalResults']
    ])
    
    return context
```

### Amazon Q Integration (24/7 Farmer Assistant)

**Amazon Q Business Application Setup**

```python
def setup_amazon_q_application():
    """
    Set up Amazon Q Business application for farmer assistance
    """
    q_business = boto3.client('qbusiness')
    
    # Create Q Business application
    app_response = q_business.create_application(
        displayName='AgriEdge Farmer Assistant',
        description='24/7 AI assistant for farmers with agricultural expertise',
        roleArn='arn:aws:iam::ACCOUNT_ID:role/AmazonQBusinessRole',
        identityCenterInstanceArn='arn:aws:sso:::instance/ssoins-XXXXX'
    )
    
    application_id = app_response['applicationId']
    
    # Create data source (S3 bucket with agricultural documents)
    data_source_response = q_business.create_data_source(
        applicationId=application_id,
        indexId='index-id',
        displayName='Agricultural Knowledge Base',
        configuration={
            's3Configuration': {
                'bucketName': 'agriedge-knowledge-base',
                'inclusionPrefixes': ['agriculture/', 'crop-diseases/', 'treatments/']
            }
        },
        roleArn='arn:aws:iam::ACCOUNT_ID:role/AmazonQDataSourceRole'
    )
    
    return application_id

def chat_with_q_assistant(user_message: str, conversation_id: str, application_id: str) -> dict:
    """
    Send message to Amazon Q assistant and get response
    """
    q_business = boto3.client('qbusiness')
    
    response = q_business.chat_sync(
        applicationId=application_id,
        conversationId=conversation_id,
        userMessage=user_message,
        userGroups=['farmers'],
        chatMode='RETRIEVAL_MODE'
    )
    
    return {
        'conversation_id': response['conversationId'],
        'message_id': response['systemMessageId'],
        'response_text': response['systemMessage'],
        'source_attributions': response.get('sourceAttributions', [])
    }

def lambda_handler(event, context):
    """
    Lambda handler for Amazon Q chat interface
    """
    body = json.loads(event['body'])
    
    user_message = body['message']
    conversation_id = body.get('conversation_id')
    user_id = body['user_id']
    language = body.get('language', 'en-IN')
    
    # Translate to English if needed (Q works best in English)
    if language != 'en-IN':
        translate = boto3.client('translate')
        translated = translate.translate_text(
            Text=user_message,
            SourceLanguageCode=language.split('-')[0],
            TargetLanguageCode='en'
        )
        user_message = translated['TranslatedText']
    
    # Get response from Q
    q_response = chat_with_q_assistant(
        user_message=user_message,
        conversation_id=conversation_id or f"conv-{user_id}-{int(time.time())}",
        application_id=os.environ['Q_APPLICATION_ID']
    )
    
    # Translate response back if needed
    response_text = q_response['response_text']
    if language != 'en-IN':
        translated_response = translate.translate_text(
            Text=response_text,
            SourceLanguageCode='en',
            TargetLanguageCode=language.split('-')[0]
        )
        response_text = translated_response['TranslatedText']
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'conversation_id': q_response['conversation_id'],
            'response': response_text,
            'sources': q_response['source_attributions']
        })
    }
```

**Android Integration with Amazon Q**

```kotlin
class AmazonQChatService(
    private val apiClient: ApiClient
) {
    
    suspend fun sendMessage(
        message: String,
        conversationId: String?,
        language: String
    ): ChatResponse {
        val request = ChatRequest(
            message = message,
            conversation_id = conversationId,
            user_id = getCurrentUserId(),
            language = language
        )
        
        return apiClient.post("/api/v1/chat/q", request)
    }
    
    suspend fun getChatHistory(conversationId: String): List<ChatMessage> {
        return apiClient.get("/api/v1/chat/q/history/$conversationId")
    }
}

data class ChatRequest(
    val message: String,
    val conversation_id: String?,
    val user_id: String,
    val language: String
)

data class ChatResponse(
    val conversation_id: String,
    val response: String,
    val sources: List<SourceAttribution>
)

data class SourceAttribution(
    val title: String,
    val snippet: String,
    val url: String?
)
```

**Primary: Amazon Transcribe and Polly**

```kotlin
class AWSVoiceService(
    private val transcribeClient: TranscribeClient,
    private val pollyClient: PollyClient,
    private val s3Client: S3Client
) {
    
    suspend fun speechToText(
        audioData: ByteArray,
        languageCode: String  // hi-IN, mr-IN, ta-IN, te-IN, kn-IN, bn-IN
    ): SpeechToTextResponse {
        // Upload audio to S3
        val audioKey = "audio/${UUID.randomUUID()}.wav"
        s3Client.putObject {
            bucket = "agriedge-audio"
            key = audioKey
            body = ByteStream.fromBytes(audioData)
        }
        
        // Start transcription job
        val jobName = "transcribe-${UUID.randomUUID()}"
        transcribeClient.startTranscriptionJob {
            transcriptionJobName = jobName
            languageCode = languageCode
            mediaFormat = MediaFormat.Wav
            media {
                mediaFileUri = "s3://agriedge-audio/$audioKey"
            }
        }
        
        // Poll for completion
        val result = waitForTranscriptionComplete(jobName)
        
        return SpeechToTextResponse(
            transcript = result.transcript,
            confidence = result.confidence
        )
    }
    
    suspend fun textToSpeech(
        text: String,
        languageCode: String,
        voiceId: String = getVoiceForLanguage(languageCode)
    ): TextToSpeechResponse {
        val response = pollyClient.synthesizeSpeech {
            this.text = text
            this.voiceId = VoiceId.fromValue(voiceId)
            this.outputFormat = OutputFormat.Mp3
            this.languageCode = languageCode
        }
        
        return TextToSpeechResponse(
            audioContent = response.audioStream?.readBytes()?.encodeBase64() ?: "",
            audioFormat = "mp3"
        )
    }
    
    private fun getVoiceForLanguage(languageCode: String): String {
        return when (languageCode) {
            "hi-IN" -> "Aditi"  // Hindi
            "mr-IN" -> "Aditi"  // Marathi (uses Hindi voice)
            "ta-IN" -> "Aditi"  // Tamil (neural voice)
            "te-IN" -> "Aditi"  // Telugu (neural voice)
            "kn-IN" -> "Aditi"  // Kannada (neural voice)
            "bn-IN" -> "Aditi"  // Bengali (neural voice)
            else -> "Aditi"
        }
    }
}

data class SpeechToTextResponse(
    val transcript: String,
    val confidence: Float
)

data class TextToSpeechResponse(
    val audioContent: String,  // Base64 encoded audio
    val audioFormat: String
)
```

**Fallback: Bhashini API**

```kotlin
interface BhashiniService {
    
    @POST("v1/speech-to-text")
    suspend fun speechToText(
        @Body request: BhashiniSpeechToTextRequest
    ): BhashiniSpeechToTextResponse
    
    @POST("v1/text-to-speech")
    suspend fun textToSpeech(
        @Body request: BhashiniTextToSpeechRequest
    ): BhashiniTextToSpeechResponse
}

// Bhashini request/response models
data class BhashiniSpeechToTextRequest(
    val audioContent: String,
    val languageCode: String,
    val encoding: AudioEncoding,
    val sampleRateHertz: Int
)

data class BhashiniSpeechToTextResponse(
    val transcript: String,
    val confidence: Float,
    val alternatives: List<Alternative>
)
```

**Voice Service with Fallback**

```kotlin
class VoiceInterfaceManager(
    private val awsVoiceService: AWSVoiceService,
    private val bhashiniService: BhashiniService
) {
    
    suspend fun speechToText(
        audioData: ByteArray,
        languageCode: String
    ): SpeechToTextResponse {
        return try {
            // Try AWS Transcribe first
            awsVoiceService.speechToText(audioData, languageCode)
        } catch (e: Exception) {
            // Fallback to Bhashini
            Log.w("VoiceInterface", "AWS Transcribe failed, using Bhashini fallback", e)
            val bhashiniResponse = bhashiniService.speechToText(
                BhashiniSpeechToTextRequest(
                    audioContent = audioData.encodeBase64(),
                    languageCode = languageCode,
                    encoding = AudioEncoding.LINEAR16,
                    sampleRateHertz = 16000
                )
            )
            SpeechToTextResponse(
                transcript = bhashiniResponse.transcript,
                confidence = bhashiniResponse.confidence
            )
        }
    }
    
    suspend fun textToSpeech(
        text: String,
        languageCode: String
    ): TextToSpeechResponse {
        return try {
            // Try AWS Polly first
            awsVoiceService.textToSpeech(text, languageCode)
        } catch (e: Exception) {
            // Fallback to Bhashini
            Log.w("VoiceInterface", "AWS Polly failed, using Bhashini fallback", e)
            val bhashiniResponse = bhashiniService.textToSpeech(
                BhashiniTextToSpeechRequest(
                    text = text,
                    languageCode = languageCode,
                    voiceGender = VoiceGender.FEMALE,
                    speakingRate = 1.0f
                )
            )
            TextToSpeechResponse(
                audioContent = bhashiniResponse.audioContent,
                audioFormat = "mp3"
            )
        }
    }
}
```

#### Voice Command Parser

```kotlin
class VoiceCommandParser(
    private val context: Context
) {
    
    fun parseCommand(transcript: String, language: String): VoiceCommand? {
        // Use pattern matching for common commands
        return when {
            matchesDiagnosisCommand(transcript, language) -> {
                VoiceCommand.StartDiagnosis(
                    cropType = extractCropType(transcript, language)
                )
            }
            matchesMarketSearchCommand(transcript, language) -> {
                VoiceCommand.SearchMarket(
                    intent = extractMarketIntent(transcript, language),
                    cropType = extractCropType(transcript, language),
                    quantity = extractQuantity(transcript, language)
                )
            }
            matchesHistoryCommand(transcript, language) -> {
                VoiceCommand.ViewHistory
            }
            else -> null
        }
    }
    
    private fun matchesDiagnosisCommand(text: String, lang: String): Boolean {
        val patterns = when (lang) {
            "hi-IN" -> listOf("रोग की जांच", "पत्ती की जांच", "फसल देखो")
            "mr-IN" -> listOf("रोग तपासा", "पान तपासा")
            "ta-IN" -> listOf("நோய் பரிசோதனை", "இலை பரிசோதனை")
            else -> listOf("diagnose", "check disease", "scan leaf")
        }
        return patterns.any { text.contains(it, ignoreCase = true) }
    }
    
    private fun extractCropType(text: String, lang: String): CropType? {
        // Extract crop mentions from text
        // Use language-specific crop name dictionaries
    }
    
    private fun extractQuantity(text: String, lang: String): Quantity? {
        // Extract numeric quantities with units
        // Handle both "100 किलो" and "1 क्विंटल"
    }
}

sealed class VoiceCommand {
    data class StartDiagnosis(val cropType: CropType?) : VoiceCommand()
    data class SearchMarket(
        val intent: MarketIntent,
        val cropType: CropType?,
        val quantity: Quantity?
    ) : VoiceCommand()
    object ViewHistory : VoiceCommand()
    object Unknown : VoiceCommand()
}

enum class MarketIntent {
    SELL_CROP, FIND_COLD_STORAGE, RENT_EQUIPMENT
}
```

#### Audio Recording Manager

```kotlin
class AudioRecorder(private val context: Context) {
    
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    
    fun startRecording(onAudioData: (ByteArray) -> Unit) {
        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )
        
        audioRecord?.startRecording()
        isRecording = true
        
        // Read audio data in background thread
        CoroutineScope(Dispatchers.IO).launch {
            val buffer = ByteArray(bufferSize)
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, bufferSize) ?: 0
                if (read > 0) {
                    onAudioData(buffer.copyOf(read))
                }
            }
        }
    }
    
    fun stopRecording(): ByteArray {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        // Return complete audio data
        return ByteArray(0) // Accumulated audio
    }
    
    companion object {
        const val SAMPLE_RATE = 16000
    }
}
```

#### Offline Voice Fallback

When Bhashini API is unavailable:
1. Display text input field automatically
2. Show notification: "Voice services unavailable, using text mode"
3. Maintain all functionality through text interface
4. Retry voice services on next network availability

### Local Database Schema

#### Room Database Definition

```kotlin
@Database(
    entities = [
        DiagnosisEntity::class,
        UserProfileEntity::class,
        TreatmentEntity::class,
        TransactionEntity::class,
        ProviderRatingEntity::class,
        SyncQueueItem::class
    ],
    version = 1,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class AgriEdgeDatabase : RoomDatabase() {
    abstract fun diagnosisDao(): DiagnosisDao
    abstract fun userProfileDao(): UserProfileDao
    abstract fun treatmentDao(): TreatmentDao
    abstract fun transactionDao(): TransactionDao
    abstract fun ratingDao(): ProviderRatingDao
    abstract fun syncQueueDao(): SyncQueueDao
}
```

#### Entity Definitions

```kotlin
@Entity(tableName = "diagnoses")
data class DiagnosisEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val userId: String,
    val timestamp: Long,
    val cropType: String,
    val diseaseId: String,
    val diseaseName: String,
    val diseaseNameLocal: String,
    val confidence: Float,
    val imagePath: String,
    val latitude: Double?,
    val longitude: Double?,
    val synced: Boolean = false
)

@Entity(tableName = "user_profile")
data class UserProfileEntity(
    @PrimaryKey
    val userId: String,
    val phoneNumber: String,
    val languageCode: String,
    val defaultLocation: String,
    val district: String,
    val state: String,
    val primaryCrops: List<String>,
    val createdAt: Long,
    val lastSyncedAt: Long?
)

@Entity(tableName = "treatments")
data class TreatmentEntity(
    @PrimaryKey
    val id: String,
    val diseaseId: String,
    val cropType: String,
    val treatmentType: String,  // ORGANIC, CHEMICAL
    val description: String,
    val descriptionLocal: String,
    val products: List<Product>,
    val applicationTiming: String,
    val dosage: String,
    val languageCode: String
)

data class Product(
    val name: String,
    val nameLocal: String,
    val type: String,
    val availability: String
)

@Entity(tableName = "transactions")
data class TransactionEntity(
    @PrimaryKey
    val id: String,
    val userId: String,
    val transactionType: String,  // SALE, COLD_STORAGE, EQUIPMENT_RENTAL
    val providerId: String,
    val providerName: String,
    val cropType: String?,
    val quantity: Double?,
    val unit: String?,
    val pricePerUnit: Double?,
    val totalAmount: Double,
    val status: String,  // INITIATED, CONFIRMED, COMPLETED, CANCELLED
    val pickupDate: Long?,
    val pickupLocation: String?,
    val createdAt: Long,
    val completedAt: Long?,
    val synced: Boolean = false
)

@Entity(tableName = "provider_ratings")
data class ProviderRatingEntity(
    @PrimaryKey
    val id: String = UUID.randomUUID().toString(),
    val userId: String,
    val providerId: String,
    val transactionId: String,
    val rating: Int,  // 1-5
    val reviewText: String?,
    val createdAt: Long,
    val synced: Boolean = false
)
```

#### Data Access Objects (DAOs)

```kotlin
@Dao
interface DiagnosisDao {
    
    @Query("SELECT * FROM diagnoses WHERE userId = :userId ORDER BY timestamp DESC")
    fun getAllDiagnoses(userId: String): Flow<List<DiagnosisEntity>>
    
    @Query("SELECT * FROM diagnoses WHERE id = :id")
    suspend fun getDiagnosisById(id: String): DiagnosisEntity?
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertDiagnosis(diagnosis: DiagnosisEntity)
    
    @Query("SELECT * FROM diagnoses WHERE synced = 0")
    suspend fun getUnsyncedDiagnoses(): List<DiagnosisEntity>
    
    @Query("UPDATE diagnoses SET synced = 1 WHERE id = :id")
    suspend fun markAsSynced(id: String)
    
    @Query("DELETE FROM diagnoses WHERE timestamp < :cutoffTime")
    suspend fun deleteOldDiagnoses(cutoffTime: Long)
}

@Dao
interface UserProfileDao {
    
    @Query("SELECT * FROM user_profile WHERE userId = :userId")
    fun getUserProfile(userId: String): Flow<UserProfileEntity?>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUserProfile(profile: UserProfileEntity)
    
    @Update
    suspend fun updateUserProfile(profile: UserProfileEntity)
}

@Dao
interface TransactionDao {
    
    @Query("SELECT * FROM transactions WHERE userId = :userId ORDER BY createdAt DESC")
    fun getAllTransactions(userId: String): Flow<List<TransactionEntity>>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTransaction(transaction: TransactionEntity)
    
    @Query("UPDATE transactions SET status = :status WHERE id = :id")
    suspend fun updateTransactionStatus(id: String, status: String)
}
```


### UI/UX Flow

#### Diagnosis Flow

```
Launch App
    │
    ▼
[Language Selection] (First launch only)
    │
    ▼
[Home Screen]
    ├─► [Voice Button] ──► [Voice Command] ──► Parse ──► Route to feature
    ├─► [Diagnose Button]
    ├─► [Market Button]
    └─► [History Button]
    │
    ▼ (Diagnose selected)
[Crop Type Selection]
    │
    ▼
[Camera Screen]
    ├─► Real-time guidance overlay
    ├─► Capture button
    └─► Gallery import option
    │
    ▼ (Image captured)
[Image Quality Check]
    ├─► Valid ──► Continue
    └─► Invalid ──► Show guidance ──► Retake
    │
    ▼
[Processing Screen] (Loading indicator)
    │
    ▼
[Diagnosis Result Screen]
    ├─► Disease name (local + scientific)
    ├─► Confidence score
    ├─► Top 3 predictions
    ├─► Treatment recommendations
    ├─► Voice readout option
    └─► Save to history (automatic)
    │
    ▼
[Treatment Details Screen]
    ├─► Organic options
    ├─► Chemical options
    ├─► Application timing
    ├─► Dosage information
    └─► Local product names
```

#### Market Flow

```
[Home Screen]
    │
    ▼
[Market Screen]
    ├─► [Sell Produce]
    ├─► [Find Cold Storage]
    └─► [Rent Equipment]
    │
    ▼ (Sell Produce selected)
[Search Form]
    ├─► Crop type
    ├─► Quantity
    └─► Location (auto-filled)
    │
    ▼
[Buyer Results]
    ├─► List of buyers
    ├─► Price per unit
    ├─► Distance
    ├─► Rating
    └─► Sort options
    │
    ▼ (Buyer selected)
[Buyer Details]
    ├─► Full quote
    ├─► Pickup details
    ├─► Reviews
    └─► Logistics bundling option
    │
    ▼
[Transaction Confirmation]
    ├─► Summary
    ├─► Terms
    └─► Confirm button
    │
    ▼
[Transaction Complete]
    ├─► Transaction ID
    ├─► Pickup details
    ├─► Contact information
    └─► Calendar reminder
```

## Backend Services Design

### Technology Stack

#### Core Technologies
- **Language**: Python 3.11+ with FastAPI or Node.js with TypeScript
- **Framework**: FastAPI (Python) or Express.js (Node.js)
- **API Style**: REST with JSON
- **Authentication**: JWT with refresh tokens
- **Compute**: AWS Lambda with API Gateway (serverless)

#### AWS AI/ML Services
- **Amazon Bedrock**: Claude 3 or Titan for treatment recommendations and knowledge synthesis
- **Amazon Q**: 24/7 farmer assistant chatbot with agricultural knowledge base
- **Amazon Transcribe**: Speech-to-text for voice interface (supports Indian languages)
- **Amazon Polly**: Text-to-speech for voice output (supports Indian languages)
- **Amazon SageMaker**: Custom ML model training and deployment for demand forecasting
- **Amazon Rekognition**: Image analysis and quality validation (optional enhancement)

#### Database & Storage
- **Primary Database**: Amazon DynamoDB (NoSQL) for high-scale, low-latency access
- **Relational Database**: Amazon RDS PostgreSQL for complex queries and analytics
- **Caching**: Amazon ElastiCache (Redis) for session management and API response caching
- **Object Storage**: Amazon S3 for diagnostic images and ML models
- **CDN**: Amazon CloudFront for global content delivery

#### AWS Infrastructure Services
- **Compute**: AWS Lambda for serverless functions
- **API Management**: Amazon API Gateway for REST APIs
- **Message Queue**: Amazon SQS for async processing
- **Event Bus**: Amazon EventBridge for event-driven architecture
- **Notifications**: Amazon SNS for SMS and push notifications
- **Secrets Management**: AWS Secrets Manager for API keys and credentials

#### Monitoring & Logging
- **Logging**: Amazon CloudWatch Logs
- **Metrics**: Amazon CloudWatch Metrics
- **Tracing**: AWS X-Ray for distributed tracing
- **Dashboards**: Amazon QuickSight for impact dashboards and analytics
- **Alarms**: Amazon CloudWatch Alarms

#### External Services
- **SMS**: Amazon SNS for OTP delivery
- **Push Notifications**: Amazon SNS Mobile Push
- **Email**: Amazon SES for transactional emails
- **Bhashini API**: Government multilingual AI services (external integration)

### Architecture Approach

**Serverless Architecture** using AWS Lambda and managed services:

```
┌─────────────────────────────────────────────────────────────┐
│                  Amazon API Gateway                          │
│              (REST API + WebSocket API)                      │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──────────────┬──────────────┬──────────────┬────────────┐
         │              │              │              │            │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐  ┌───▼────┐
    │  Auth   │   │  User   │   │  Sync   │   │ Market  │  │Telemetry│
    │ Lambda  │   │ Profile │   │ Lambda  │   │Connector│  │ Lambda  │
    │         │   │ Lambda  │   │         │   │ Lambda  │  │         │
    └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘  └───┬────┘
         │              │              │              │           │
         └──────────────┴──────────────┴──────────────┴───────────┘
                                  │
         ┌────────────────────────┼────────────────────────────────┐
         │                        │                                │
    ┌────▼────────┐    ┌─────────▼──────────┐    ┌──────────────▼──┐
    │  DynamoDB   │    │   RDS PostgreSQL   │    │   Amazon S3     │
    │  (NoSQL)    │    │   (Analytics)      │    │  (Images/Models)│
    └─────────────┘    └────────────────────┘    └─────────────────┘
         │
         │
    ┌────▼────────────────────────────────────────────────────────┐
    │              AWS AI/ML Services                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
    │  │   Bedrock    │  │   Amazon Q   │  │ Transcribe/Polly │ │
    │  │ (Claude/     │  │  (Chatbot)   │  │  (Voice I/O)     │ │
    │  │  Titan)      │  │              │  │                  │ │
    │  └──────────────┘  └──────────────┘  └──────────────────┘ │
    └──────────────────────────────────────────────────────────────┘
```

**Lambda Function Responsibilities**:

1. **Auth Lambda**: User authentication, OTP generation/verification via SNS, JWT management
2. **User Profile Lambda**: User data management in DynamoDB, preferences, location
3. **Sync Lambda**: Handles data synchronization from mobile devices, writes to DynamoDB/S3
4. **Market Connector Lambda**: Beckn/ONDC integration, transaction management
5. **Telemetry Lambda**: Collects and processes anonymized usage data, writes to DynamoDB
6. **Treatment Generator Lambda**: Uses Amazon Bedrock to generate treatment recommendations
7. **Voice Interface Lambda**: Integrates with Transcribe/Polly for voice I/O
8. **Q Assistant Lambda**: Handles Amazon Q chatbot interactions

### API Design

#### Authentication Endpoints

```
POST /api/v1/auth/register
Request:
{
  "phoneNumber": "+919876543210",
  "languageCode": "hi-IN"
}
Response:
{
  "userId": "uuid",
  "otpSent": true,
  "expiresIn": 600
}

POST /api/v1/auth/verify-otp
Request:
{
  "phoneNumber": "+919876543210",
  "otp": "123456"
}
Response:
{
  "userId": "uuid",
  "accessToken": "jwt-token",
  "refreshToken": "refresh-token",
  "expiresIn": 3600
}

POST /api/v1/auth/refresh
Request:
{
  "refreshToken": "refresh-token"
}
Response:
{
  "accessToken": "new-jwt-token",
  "expiresIn": 3600
}

POST /api/v1/auth/logout
Headers: Authorization: Bearer <token>
Response: 204 No Content
```

#### User Profile Endpoints

```
GET /api/v1/profile
Headers: Authorization: Bearer <token>
Response:
{
  "userId": "uuid",
  "phoneNumber": "+919876543210",
  "languageCode": "hi-IN",
  "defaultLocation": {
    "village": "Shirpur",
    "district": "Dhule",
    "state": "Maharashtra"
  },
  "primaryCrops": ["cotton", "sugarcane"],
  "createdAt": "2024-01-15T10:30:00Z",
  "lastSyncedAt": "2024-01-20T15:45:00Z"
}

PUT /api/v1/profile
Headers: Authorization: Bearer <token>
Request:
{
  "languageCode": "mr-IN",
  "defaultLocation": {
    "village": "Shirpur",
    "district": "Dhule",
    "state": "Maharashtra"
  },
  "primaryCrops": ["cotton", "sugarcane", "wheat"]
}
Response: 200 OK (returns updated profile)

DELETE /api/v1/profile
Headers: Authorization: Bearer <token>
Response: 204 No Content
```

#### Sync Endpoints

```
POST /api/v1/sync/diagnoses
Headers: Authorization: Bearer <token>
Request:
{
  "diagnoses": [
    {
      "id": "uuid",
      "timestamp": 1705320000000,
      "cropType": "cotton",
      "diseaseId": "cotton_leaf_curl",
      "diseaseName": "Cotton Leaf Curl",
      "confidence": 0.92,
      "imageUrl": "s3://bucket/images/uuid.jpg",
      "location": {
        "latitude": 21.0,
        "longitude": 74.5
      }
    }
  ]
}
Response:
{
  "synced": 1,
  "failed": 0,
  "errors": []
}

POST /api/v1/sync/ratings
Headers: Authorization: Bearer <token>
Request:
{
  "ratings": [
    {
      "id": "uuid",
      "providerId": "provider-uuid",
      "transactionId": "txn-uuid",
      "rating": 5,
      "reviewText": "Excellent service",
      "createdAt": 1705320000000
    }
  ]
}
Response:
{
  "synced": 1,
  "failed": 0
}

GET /api/v1/sync/status
Headers: Authorization: Bearer <token>
Response:
{
  "lastSyncedAt": "2024-01-20T15:45:00Z",
  "pendingItems": 0,
  "syncEnabled": true
}
```

#### Market Endpoints

```
POST /api/v1/market/search/buyers
Headers: Authorization: Bearer <token>
Request:
{
  "cropType": "wheat",
  "quantity": 100,
  "unit": "quintal",
  "location": {
    "latitude": 21.0,
    "longitude": 74.5
  },
  "radius": 50
}
Response:
{
  "buyers": [
    {
      "providerId": "uuid",
      "providerName": "ABC Traders",
      "pricePerUnit": 2500,
      "unit": "quintal",
      "totalAmount": 250000,
      "distance": 15.5,
      "rating": 4.5,
      "totalTransactions": 120,
      "pickupAvailable": true,
      "estimatedPickupDate": "2024-01-25"
    }
  ],
  "totalResults": 5
}

POST /api/v1/market/search/cold-storage
Headers: Authorization: Bearer <token>
Request:
{
  "location": {
    "latitude": 21.0,
    "longitude": 74.5
  },
  "radius": 50,
  "requiredCapacity": 10,
  "unit": "ton",
  "duration": 30
}
Response:
{
  "facilities": [
    {
      "providerId": "uuid",
      "facilityName": "Cold Chain Solutions",
      "distance": 12.3,
      "dailyRate": 50,
      "unit": "ton",
      "availableCapacity": 100,
      "rating": 4.7,
      "totalReviews": 45,
      "address": "Industrial Area, Dhule"
    }
  ],
  "totalResults": 3
}

POST /api/v1/market/search/equipment
Headers: Authorization: Bearer <token>
Request:
{
  "equipmentType": "tractor",
  "location": {
    "latitude": 21.0,
    "longitude": 74.5
  },
  "radius": 30,
  "startDate": "2024-01-25",
  "endDate": "2024-01-27"
}
Response:
{
  "equipment": [
    {
      "providerId": "uuid",
      "providerName": "Farm Equipment Rentals",
      "equipmentType": "tractor",
      "model": "Mahindra 575",
      "horsepower": 45,
      "dailyRate": 1500,
      "distance": 8.5,
      "rating": 4.6,
      "deliveryAvailable": true,
      "deliveryCharge": 500
    }
  ],
  "totalResults": 4
}

POST /api/v1/market/transactions
Headers: Authorization: Bearer <token>
Request:
{
  "transactionType": "SALE",
  "providerId": "uuid",
  "cropType": "wheat",
  "quantity": 100,
  "unit": "quintal",
  "pricePerUnit": 2500,
  "pickupDate": "2024-01-25T10:00:00Z",
  "pickupLocation": "Farm address",
  "bundleLogistics": false
}
Response:
{
  "transactionId": "uuid",
  "status": "INITIATED",
  "providerContact": "+919876543210",
  "pickupDetails": {
    "date": "2024-01-25T10:00:00Z",
    "location": "Farm address",
    "contactPerson": "Ramesh Kumar"
  },
  "totalAmount": 250000
}

GET /api/v1/market/transactions/:id
Headers: Authorization: Bearer <token>
Response:
{
  "transactionId": "uuid",
  "status": "CONFIRMED",
  "createdAt": "2024-01-20T10:00:00Z",
  "updatedAt": "2024-01-20T11:00:00Z",
  "details": { ... }
}

GET /api/v1/market/providers/:id/reviews
Response:
{
  "providerId": "uuid",
  "averageRating": 4.5,
  "totalReviews": 120,
  "ratingDistribution": {
    "5": 80,
    "4": 30,
    "3": 8,
    "2": 1,
    "1": 1
  },
  "reviews": [
    {
      "reviewId": "uuid",
      "rating": 5,
      "reviewText": "Excellent service",
      "createdAt": "2024-01-15T10:00:00Z",
      "userName": "Anonymous"
    }
  ]
}
```

#### Telemetry Endpoints

```
POST /api/v1/telemetry/events
Headers: Authorization: Bearer <token>
Request:
{
  "events": [
    {
      "eventType": "DIAGNOSIS_COMPLETED",
      "timestamp": 1705320000000,
      "metadata": {
        "cropType": "cotton",
        "diseaseId": "cotton_leaf_curl",
        "confidence": 0.92,
        "inferenceTime": 2500,
        "deviceModel": "Redmi Note 10",
        "osVersion": "Android 12"
      }
    }
  ]
}
Response: 202 Accepted
```


### Beckn/ONDC Integration Layer

#### Beckn Protocol Overview

Beckn is an open protocol for decentralized commerce. The integration involves:
1. **Discovery**: Search for providers (buyers, cold storage, equipment)
2. **Order**: Initiate transactions
3. **Fulfillment**: Track and complete transactions
4. **Post-Fulfillment**: Ratings and support

#### Beckn Client Implementation

```kotlin
// Backend service
class BecknClient(
    private val gatewayUrl: String,
    private val subscriberId: String,
    private val subscriberUrl: String,
    private val privateKey: String
) {
    
    suspend fun search(context: BecknContext, intent: Intent): SearchResponse {
        val request = SearchRequest(
            context = context,
            message = Message(intent = intent)
        )
        
        val signature = signRequest(request, privateKey)
        
        return httpClient.post("$gatewayUrl/search") {
            header("Authorization", signature)
            setBody(request)
        }
    }
    
    suspend fun select(context: BecknContext, order: Order): SelectResponse {
        val request = SelectRequest(
            context = context,
            message = Message(order = order)
        )
        
        return httpClient.post("$gatewayUrl/select") {
            header("Authorization", signRequest(request, privateKey))
            setBody(request)
        }
    }
    
    suspend fun init(context: BecknContext, order: Order): InitResponse {
        // Initialize transaction
    }
    
    suspend fun confirm(context: BecknContext, order: Order): ConfirmResponse {
        // Confirm transaction
    }
    
    suspend fun status(context: BecknContext, orderId: String): StatusResponse {
        // Get transaction status
    }
    
    suspend fun cancel(context: BecknContext, orderId: String, reason: String): CancelResponse {
        // Cancel transaction
    }
    
    suspend fun rating(context: BecknContext, rating: Rating): RatingResponse {
        // Submit rating
    }
}

data class BecknContext(
    val domain: String,  // "agriculture:1.0.0"
    val country: String,  // "IND"
    val city: String,
    val action: String,
    val coreVersion: String,  // "1.0.0"
    val bapId: String,  // Buyer app ID
    val bapUri: String,
    val transactionId: String,
    val messageId: String,
    val timestamp: String
)

data class Intent(
    val item: Item?,
    val fulfillment: Fulfillment?,
    val provider: Provider?
)

data class Item(
    val descriptor: Descriptor,
    val quantity: Quantity?,
    val category: Category?
)

data class Descriptor(
    val name: String,
    val code: String?,
    val shortDesc: String?,
    val longDesc: String?
)

data class Fulfillment(
    val type: String,  // "Pickup", "Delivery"
    val start: Location?,
    val end: Location?
)

data class Location(
    val gps: String,  // "lat,long"
    val address: Address?,
    val city: City?,
    val state: State?
)
```

#### Beckn Webhook Handler

The backend must implement webhook endpoints to receive async responses:

```kotlin
@RestController
@RequestMapping("/api/v1/beckn/webhooks")
class BecknWebhookController(
    private val transactionService: TransactionService
) {
    
    @PostMapping("/on_search")
    suspend fun onSearch(@RequestBody response: OnSearchResponse) {
        // Process search results
        // Store in cache for user retrieval
        transactionService.cacheSearchResults(
            transactionId = response.context.transactionId,
            providers = response.message.catalog.providers
        )
    }
    
    @PostMapping("/on_select")
    suspend fun onSelect(@RequestBody response: OnSelectResponse) {
        // Process quote from provider
        transactionService.updateQuote(
            transactionId = response.context.transactionId,
            quote = response.message.order.quote
        )
    }
    
    @PostMapping("/on_init")
    suspend fun onInit(@RequestBody response: OnInitResponse) {
        // Process initialized order
        transactionService.updateOrderDetails(
            transactionId = response.context.transactionId,
            order = response.message.order
        )
    }
    
    @PostMapping("/on_confirm")
    suspend fun onConfirm(@RequestBody response: OnConfirmResponse) {
        // Process confirmed order
        transactionService.confirmTransaction(
            transactionId = response.context.transactionId,
            order = response.message.order
        )
        
        // Send notification to user
        notificationService.sendTransactionConfirmation(
            userId = getUserIdFromTransaction(response.context.transactionId),
            transactionDetails = response.message.order
        )
    }
    
    @PostMapping("/on_status")
    suspend fun onStatus(@RequestBody response: OnStatusResponse) {
        // Update transaction status
        transactionService.updateStatus(
            transactionId = response.context.transactionId,
            status = response.message.order.state
        )
    }
    
    @PostMapping("/on_cancel")
    suspend fun onCancel(@RequestBody response: OnCancelResponse) {
        // Handle cancellation
        transactionService.cancelTransaction(
            transactionId = response.context.transactionId,
            reason = response.message.order.cancellationReasons
        )
    }
}
```

### Database Schema

#### DynamoDB Tables

**Users Table**

```python
# Table: Users
# Partition Key: user_id (String)

{
    "user_id": "uuid",
    "phone_number": "string",  # GSI partition key
    "phone_verified": "boolean",
    "language_code": "string",
    "created_at": "number",  # Unix timestamp
    "updated_at": "number",
    "last_login_at": "number",
    "deleted_at": "number"  # Soft delete
}

# Global Secondary Index: PhoneNumberIndex
# Partition Key: phone_number
```

**User Profiles Table**

```python
# Table: UserProfiles
# Partition Key: user_id (String)

{
    "user_id": "uuid",
    "village": "string",
    "district": "string",
    "state": "string",
    "latitude": "number",
    "longitude": "number",
    "primary_crops": ["string"],  # List of crop types
    "last_synced_at": "number",
    "created_at": "number",
    "updated_at": "number"
}

# Global Secondary Index: LocationIndex
# Partition Key: state
# Sort Key: district
```

**Diagnoses Table**

```python
# Table: Diagnoses
# Partition Key: user_id (String)
# Sort Key: timestamp (Number)

{
    "user_id": "uuid",
    "timestamp": "number",  # Unix timestamp (sort key)
    "diagnosis_id": "uuid",  # GSI partition key
    "crop_type": "string",
    "disease_id": "string",
    "disease_name": "string",
    "confidence": "number",
    "image_url": "string",  # S3 URL
    "latitude": "number",
    "longitude": "number",
    "synced_at": "number"
}

# Global Secondary Index: DiagnosisIdIndex
# Partition Key: diagnosis_id

# Global Secondary Index: DiseaseIndex
# Partition Key: disease_id
# Sort Key: timestamp
```

**Transactions Table**

```python
# Table: Transactions
# Partition Key: user_id (String)
# Sort Key: created_at (Number)

{
    "user_id": "uuid",
    "created_at": "number",  # Sort key
    "transaction_id": "uuid",  # GSI partition key
    "beckn_transaction_id": "string",
    "transaction_type": "string",  # SALE, COLD_STORAGE, EQUIPMENT_RENTAL
    "provider_id": "string",
    "provider_name": "string",
    "crop_type": "string",
    "quantity": "number",
    "unit": "string",
    "price_per_unit": "number",
    "total_amount": "number",
    "status": "string",  # INITIATED, CONFIRMED, IN_PROGRESS, COMPLETED, CANCELLED
    "pickup_date": "number",
    "pickup_location": "string",
    "updated_at": "number",
    "completed_at": "number"
}

# Global Secondary Index: TransactionIdIndex
# Partition Key: transaction_id

# Global Secondary Index: StatusIndex
# Partition Key: status
# Sort Key: created_at
```

**Provider Ratings Table**

```python
# Table: ProviderRatings
# Partition Key: provider_id (String)
# Sort Key: created_at (Number)

{
    "provider_id": "string",
    "created_at": "number",  # Sort key
    "rating_id": "uuid",
    "user_id": "uuid",
    "transaction_id": "uuid",
    "rating": "number",  # 1-5
    "review_text": "string",
    "synced_at": "number"
}

# Global Secondary Index: UserRatingsIndex
# Partition Key: user_id
# Sort Key: created_at
```

**Telemetry Events Table**

```python
# Table: TelemetryEvents
# Partition Key: event_type (String)
# Sort Key: timestamp (Number)

{
    "event_type": "string",  # Partition key
    "timestamp": "number",  # Sort key
    "event_id": "uuid",
    "user_hash": "string",  # Anonymized user ID
    "device_model": "string",
    "os_version": "string",
    "app_version": "string",
    "metadata": {
        "crop_type": "string",
        "disease_id": "string",
        "confidence": "number",
        "inference_time": "number"
    },
    "created_at": "number"
}

# Time-to-Live (TTL) enabled on 'ttl' attribute (90 days retention)
```

**Sync Queue Table**

```python
# Table: SyncQueue
# Partition Key: user_id (String)
# Sort Key: timestamp (Number)

{
    "user_id": "uuid",
    "timestamp": "number",  # Sort key
    "sync_id": "uuid",
    "entity_type": "string",  # DIAGNOSIS, PROFILE, RATING, TRANSACTION
    "entity_id": "string",
    "operation": "string",  # CREATE, UPDATE, DELETE
    "payload": "map",  # JSON object
    "retry_count": "number",
    "status": "string",  # PENDING, IN_PROGRESS, FAILED, COMPLETED
    "last_attempt_at": "number"
}

# Global Secondary Index: StatusIndex
# Partition Key: status
# Sort Key: timestamp
```

**Treatment Recommendations Table (Cache)**

```python
# Table: TreatmentRecommendations
# Partition Key: disease_id (String)
# Sort Key: language (String)

{
    "disease_id": "string",
    "language": "string",  # Sort key
    "crop_type": "string",
    "treatment_text": "string",  # Generated by Bedrock
    "organic_options": ["map"],
    "chemical_options": ["map"],
    "preventive_measures": ["string"],
    "generated_at": "number",
    "ttl": "number"  # TTL for cache expiration (30 days)
}
```

**Beckn Search Cache Table**

```python
# Table: BecknSearchCache
# Partition Key: transaction_id (String)

{
    "transaction_id": "string",
    "user_id": "uuid",
    "search_type": "string",  # BUYER, COLD_STORAGE, EQUIPMENT
    "results": ["map"],  # Array of provider results
    "created_at": "number",
    "ttl": "number"  # TTL for cache expiration (1 hour)
}
```

#### RDS PostgreSQL Schema (Analytics)

For complex analytics and reporting, use RDS PostgreSQL:

```sql
-- Aggregated diagnoses for analytics
CREATE TABLE diagnosis_analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    crop_type VARCHAR(50),
    disease_id VARCHAR(100),
    total_diagnoses INT,
    avg_confidence DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_diagnosis_analytics_date ON diagnosis_analytics(date);
CREATE INDEX idx_diagnosis_analytics_location ON diagnosis_analytics(state, district);

-- Transaction analytics
CREATE TABLE transaction_analytics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    transaction_type VARCHAR(50),
    total_transactions INT,
    total_value DECIMAL(12, 2),
    avg_transaction_value DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Provider performance metrics
CREATE TABLE provider_metrics (
    provider_id VARCHAR(100) PRIMARY KEY,
    provider_name VARCHAR(200),
    provider_type VARCHAR(50),
    total_transactions INT,
    avg_rating DECIMAL(3, 2),
    total_reviews INT,
    rating_distribution JSONB,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Authentication and Authorization

#### JWT Token Structure

```json
{
  "sub": "user-uuid",
  "phone": "+919876543210",
  "iat": 1705320000,
  "exp": 1705323600,
  "roles": ["FARMER"]
}
```

#### Authentication Flow

```
Mobile App                    Backend                      SMS Gateway
    │                            │                              │
    │  POST /auth/register       │                              │
    │  {phoneNumber}             │                              │
    ├───────────────────────────►│                              │
    │                            │  Generate OTP                │
    │                            │  Store in DB                 │
    │                            │  Send SMS                    │
    │                            ├─────────────────────────────►│
    │                            │                              │
    │  {otpSent: true}           │                              │
    │◄───────────────────────────┤                              │
    │                            │                              │
    │  POST /auth/verify-otp     │                              │
    │  {phoneNumber, otp}        │                              │
    ├───────────────────────────►│                              │
    │                            │  Verify OTP                  │
    │                            │  Create user if new          │
    │                            │  Generate JWT tokens         │
    │                            │                              │
    │  {accessToken, refreshToken}                              │
    │◄───────────────────────────┤                              │
    │                            │                              │
    │  Store tokens locally      │                              │
    │                            │                              │
    │  Subsequent requests       │                              │
    │  Authorization: Bearer <token>                            │
    ├───────────────────────────►│                              │
    │                            │  Verify JWT                  │
    │                            │  Process request             │
    │  Response                  │                              │
    │◄───────────────────────────┤                              │
```

#### Security Implementation

```kotlin
@Configuration
@EnableWebSecurity
class SecurityConfig {
    
    @Bean
    fun securityFilterChain(http: HttpSecurity): SecurityFilterChain {
        http
            .csrf { it.disable() }
            .cors { it.configurationSource(corsConfigurationSource()) }
            .authorizeHttpRequests { auth ->
                auth
                    .requestMatchers("/api/v1/auth/**").permitAll()
                    .requestMatchers("/api/v1/beckn/webhooks/**").permitAll()
                    .anyRequest().authenticated()
            }
            .sessionManagement { it.sessionCreationPolicy(SessionCreationPolicy.STATELESS) }
            .addFilterBefore(jwtAuthenticationFilter(), UsernamePasswordAuthenticationFilter::class.java)
        
        return http.build()
    }
    
    @Bean
    fun jwtAuthenticationFilter(): JwtAuthenticationFilter {
        return JwtAuthenticationFilter(jwtService)
    }
}

class JwtAuthenticationFilter(
    private val jwtService: JwtService
) : OncePerRequestFilter() {
    
    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain
    ) {
        val token = extractToken(request)
        
        if (token != null && jwtService.validateToken(token)) {
            val userId = jwtService.getUserIdFromToken(token)
            val authentication = UsernamePasswordAuthenticationToken(
                userId, null, emptyList()
            )
            SecurityContextHolder.getContext().authentication = authentication
        }
        
        filterChain.doFilter(request, response)
    }
    
    private fun extractToken(request: HttpServletRequest): String? {
        val bearerToken = request.getHeader("Authorization")
        return if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            bearerToken.substring(7)
        } else null
    }
}
```


### Cloud Storage Strategy (AWS S3 + CloudFront)

#### Image Storage

**S3 Bucket Structure**:
```
s3://agriedge-images/
├── diagnoses/
│   ├── {user_id}/
│   │   ├── {year}/
│   │   │   ├── {month}/
│   │   │   │   ├── {diagnosis_id}.jpg
│   │   │   │   └── {diagnosis_id}_thumb.jpg
```

**S3 Bucket Configuration**:
```python
import boto3

s3_client = boto3.client('s3')

# Create bucket with lifecycle policies
s3_client.create_bucket(
    Bucket='agriedge-images',
    CreateBucketConfiguration={'LocationConstraint': 'ap-south-1'}  # Mumbai region
)

# Enable versioning
s3_client.put_bucket_versioning(
    Bucket='agriedge-images',
    VersioningConfiguration={'Status': 'Enabled'}
)

# Set lifecycle policy (move to Glacier after 90 days)
lifecycle_policy = {
    'Rules': [
        {
            'Id': 'MoveToGlacier',
            'Status': 'Enabled',
            'Transitions': [
                {
                    'Days': 90,
                    'StorageClass': 'GLACIER'
                }
            ],
            'NoncurrentVersionTransitions': [
                {
                    'NoncurrentDays': 30,
                    'StorageClass': 'GLACIER'
                }
            ]
        }
    ]
}

s3_client.put_bucket_lifecycle_configuration(
    Bucket='agriedge-images',
    LifecycleConfiguration=lifecycle_policy
)

# Enable server-side encryption
s3_client.put_bucket_encryption(
    Bucket='agriedge-images',
    ServerSideEncryptionConfiguration={
        'Rules': [
            {
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'AES256'
                }
            }
        ]
    }
)
```

**Upload Flow with Presigned URLs**:
1. Mobile app requests presigned URL from Lambda
2. Lambda generates presigned URL (15-minute expiration)
3. App uploads directly to S3 using presigned URL
4. App sends diagnosis metadata to API with S3 URL

**Presigned URL Generation (Lambda)**:
```python
import boto3
from datetime import timedelta

s3_client = boto3.client('s3')

def generate_presigned_upload_url(user_id: str, diagnosis_id: str) -> dict:
    """
    Generate presigned URL for image upload
    """
    key = f"diagnoses/{user_id}/{datetime.now().year}/{datetime.now().month}/{diagnosis_id}.jpg"
    
    presigned_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': 'agriedge-images',
            'Key': key,
            'ContentType': 'image/jpeg'
        },
        ExpiresIn=900  # 15 minutes
    )
    
    return {
        'upload_url': presigned_url,
        'key': key,
        'expires_in': 900
    }

def lambda_handler(event, context):
    body = json.loads(event['body'])
    user_id = body['user_id']
    diagnosis_id = body['diagnosis_id']
    
    result = generate_presigned_upload_url(user_id, diagnosis_id)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

**CloudFront Distribution**:
```python
cloudfront_client = boto3.client('cloudfront')

# Create CloudFront distribution for S3 bucket
distribution_config = {
    'CallerReference': str(time.time()),
    'Comment': 'AgriEdge Images CDN',
    'Enabled': True,
    'Origins': {
        'Quantity': 1,
        'Items': [
            {
                'Id': 'S3-agriedge-images',
                'DomainName': 'agriedge-images.s3.ap-south-1.amazonaws.com',
                'S3OriginConfig': {
                    'OriginAccessIdentity': 'origin-access-identity/cloudfront/XXXXX'
                }
            }
        ]
    },
    'DefaultCacheBehavior': {
        'TargetOriginId': 'S3-agriedge-images',
        'ViewerProtocolPolicy': 'redirect-to-https',
        'AllowedMethods': {
            'Quantity': 2,
            'Items': ['GET', 'HEAD']
        },
        'Compress': True,
        'MinTTL': 0,
        'DefaultTTL': 86400,  # 1 day
        'MaxTTL': 31536000  # 1 year
    },
    'PriceClass': 'PriceClass_200'  # Use edge locations in India and Asia
}
```

#### ML Model Distribution

**Model Storage**:
```
s3://agriedge-models/
├── production/
│   ├── crop_disease_classifier_v1.0.0.tflite
│   ├── crop_disease_classifier_v1.0.0.json  (metadata)
│   └── crop_disease_classifier_v1.0.0.sha256
├── staging/
│   └── ...
```

**Model Metadata**:
```json
{
  "version": "1.0.0",
  "modelType": "crop_disease_classifier",
  "checksum": "sha256-hash",
  "size": 45678901,
  "minAppVersion": "1.0.0",
  "supportedCrops": ["rice", "wheat", "cotton", "tomato", "potato", "sugarcane"],
  "diseaseCount": 42,
  "accuracy": 0.87,
  "releaseDate": "2024-01-15T00:00:00Z",
  "downloadUrl": "https://cdn.agriedge.com/models/v1.0.0/model.tflite",
  "releaseNotes": "Initial release with 42 disease classes"
}
```

**Model Update API (Lambda)**:
```python
def lambda_handler(event, context):
    """
    Get latest model metadata
    """
    s3_client = boto3.client('s3')
    
    # Get latest model metadata from S3
    response = s3_client.get_object(
        Bucket='agriedge-models',
        Key='production/latest.json'
    )
    
    metadata = json.loads(response['Body'].read())
    
    # Generate CloudFront URL for download
    metadata['downloadUrl'] = f"https://cdn.agriedge.com/models/{metadata['version']}/model.tflite"
    
    return {
        'statusCode': 200,
        'body': json.dumps(metadata)
    }
```

### AWS Cost Estimation

**Monthly Cost Breakdown (Moderate Scale: 10,000 active users)**

#### Compute Costs
- **AWS Lambda**:
  - Requests: ~5M requests/month
  - Duration: Avg 500ms, 512MB memory
  - Cost: ~$25/month
  
- **API Gateway**:
  - REST API calls: ~5M requests/month
  - Cost: ~$17.50/month

#### Storage Costs
- **DynamoDB**:
  - On-Demand pricing
  - Storage: ~50GB
  - Read/Write: ~10M reads, ~2M writes/month
  - Cost: ~$150/month

- **S3**:
  - Standard storage: ~500GB (images)
  - Glacier storage: ~2TB (old images)
  - Requests: ~1M PUT, ~5M GET/month
  - Cost: ~$30/month (Standard) + ~$8/month (Glacier)

- **RDS PostgreSQL** (db.t3.medium):
  - Instance: ~$60/month
  - Storage: 100GB SSD
  - Cost: ~$75/month

#### AI/ML Costs
- **Amazon Bedrock** (Claude 3 Sonnet):
  - Input tokens: ~5M tokens/month
  - Output tokens: ~2M tokens/month
  - Cost: ~$60/month

- **Amazon Q Business**:
  - 100 users (farmers + support staff)
  - Cost: ~$200/month

- **Amazon Transcribe**:
  - ~10,000 minutes/month
  - Cost: ~$24/month

- **Amazon Polly**:
  - ~5M characters/month
  - Cost: ~$20/month

#### Data Transfer & CDN
- **CloudFront**:
  - Data transfer: ~1TB/month
  - Requests: ~10M/month
  - Cost: ~$85/month

- **Data Transfer Out**:
  - ~500GB/month
  - Cost: ~$45/month

#### Monitoring & Other Services
- **CloudWatch**:
  - Logs: ~50GB/month
  - Metrics: Custom metrics
  - Cost: ~$15/month

- **SNS** (SMS for OTP):
  - ~5,000 SMS/month
  - Cost: ~$25/month

- **ElastiCache** (Redis):
  - cache.t3.micro
  - Cost: ~$15/month

**Total Estimated Monthly Cost: ~$854/month**

**Cost Optimization Strategies**:
1. Use Lambda reserved concurrency for predictable workloads
2. Enable S3 Intelligent-Tiering for automatic cost optimization
3. Use DynamoDB reserved capacity for base load
4. Implement aggressive caching with CloudFront and ElastiCache
5. Use Spot Instances for batch processing (SageMaker training)
6. Enable S3 Transfer Acceleration only for critical uploads
7. Use AWS Savings Plans for 1-year commitment (30% savings)

**Estimated Cost at Scale (100,000 users): ~$4,500/month**

### Data Synchronization Service

#### Sync Service Architecture

```kotlin
@Service
class SyncService(
    private val diagnosisRepository: DiagnosisRepository,
    private val ratingRepository: RatingRepository,
    private val telemetryRepository: TelemetryRepository,
    private val userProfileRepository: UserProfileRepository
) {
    
    suspend fun syncDiagnoses(userId: String, diagnoses: List<DiagnosisDto>): SyncResult {
        val results = diagnoses.map { dto ->
            try {
                val entity = dto.toEntity(userId)
                diagnosisRepository.save(entity)
                SyncItemResult.Success(dto.id)
            } catch (e: Exception) {
                SyncItemResult.Failure(dto.id, e.message ?: "Unknown error")
            }
        }
        
        return SyncResult(
            synced = results.count { it is SyncItemResult.Success },
            failed = results.count { it is SyncItemResult.Failure },
            errors = results.filterIsInstance<SyncItemResult.Failure>()
        )
    }
    
    suspend fun syncRatings(userId: String, ratings: List<RatingDto>): SyncResult {
        // Similar implementation
        // Also forward ratings to Beckn network
    }
    
    suspend fun syncTelemetry(userId: String, events: List<TelemetryEventDto>): SyncResult {
        // Anonymize data before storage
        val anonymizedEvents = events.map { it.anonymize() }
        
        // Store in database
        telemetryRepository.saveAll(anonymizedEvents.map { it.toEntity() })
        
        return SyncResult(synced = events.size, failed = 0, errors = emptyList())
    }
}

data class SyncResult(
    val synced: Int,
    val failed: Int,
    val errors: List<SyncItemResult.Failure>
)

sealed class SyncItemResult {
    data class Success(val id: String) : SyncItemResult()
    data class Failure(val id: String, val error: String) : SyncItemResult()
}
```

#### Conflict Resolution

**Strategy**: Last Write Wins (LWW) with server timestamp

```kotlin
class ConflictResolver {
    
    fun resolveUserProfile(
        local: UserProfileEntity,
        remote: UserProfileEntity
    ): UserProfileEntity {
        // Server timestamp is source of truth
        return if (remote.updatedAt > local.updatedAt) {
            remote
        } else {
            local
        }
    }
    
    fun resolveDiagnosis(
        local: DiagnosisEntity,
        remote: DiagnosisEntity
    ): DiagnosisEntity {
        // Diagnoses are immutable, use creation timestamp
        return if (remote.timestamp > local.timestamp) {
            remote
        } else {
            local
        }
    }
}
```

## Data Models

### Domain Models (Shared Concepts)

#### Diagnosis

```kotlin
data class Diagnosis(
    val id: String,
    val userId: String,
    val timestamp: Instant,
    val cropType: CropType,
    val disease: Disease,
    val confidence: Float,
    val image: DiagnosisImage,
    val location: GeoLocation?,
    val synced: Boolean
)

data class Disease(
    val id: String,
    val name: String,
    val localName: String,
    val scientificName: String,
    val category: DiseaseCategory
)

enum class DiseaseCategory {
    FUNGAL, BACTERIAL, VIRAL, PEST, NUTRIENT_DEFICIENCY
}

data class DiagnosisImage(
    val localPath: String?,
    val remoteUrl: String?,
    val thumbnailUrl: String?
)

data class GeoLocation(
    val latitude: Double,
    val longitude: Double
)
```

#### Treatment

```kotlin
data class Treatment(
    val id: String,
    val diseaseId: String,
    val cropType: CropType,
    val type: TreatmentType,
    val description: String,
    val localizedDescription: String,
    val products: List<Product>,
    val applicationTiming: String,
    val dosage: String
)

enum class TreatmentType {
    ORGANIC, CHEMICAL, CULTURAL, BIOLOGICAL
}

data class Product(
    val name: String,
    val localName: String,
    val type: ProductType,
    val activeIngredient: String?,
    val availability: Availability
)

enum class ProductType {
    PESTICIDE, FUNGICIDE, INSECTICIDE, FERTILIZER, BIO_AGENT
}

enum class Availability {
    WIDELY_AVAILABLE, REGIONAL, LIMITED, PRESCRIPTION_REQUIRED
}
```

#### Market Entities

```kotlin
data class BuyerQuote(
    val providerId: String,
    val providerName: String,
    val pricePerUnit: Money,
    val totalAmount: Money,
    val distance: Distance,
    val rating: ProviderRating,
    val pickupDetails: PickupDetails?,
    val logisticsBundling: LogisticsBundling?
)

data class Money(
    val amount: BigDecimal,
    val currency: String = "INR"
)

data class Distance(
    val value: Double,
    val unit: DistanceUnit = DistanceUnit.KILOMETERS
)

enum class DistanceUnit {
    KILOMETERS, MILES
}

data class ProviderRating(
    val average: Float,
    val totalReviews: Int,
    val distribution: Map<Int, Int>
)

data class PickupDetails(
    val date: Instant,
    val location: String,
    val contactPerson: String,
    val contactPhone: String
)

data class LogisticsBundling(
    val available: Boolean,
    val transportCost: Money?,
    val estimatedPickupTime: Instant?
)

data class ColdStorageFacility(
    val providerId: String,
    val facilityName: String,
    val distance: Distance,
    val dailyRate: Money,
    val availableCapacity: Capacity,
    val rating: ProviderRating,
    val address: String,
    val amenities: List<String>
)

data class Capacity(
    val value: Double,
    val unit: CapacityUnit
)

enum class CapacityUnit {
    TON, CUBIC_METER, QUINTAL
}

data class EquipmentRental(
    val providerId: String,
    val providerName: String,
    val equipmentType: EquipmentType,
    val model: String,
    val specifications: EquipmentSpecs,
    val dailyRate: Money,
    val distance: Distance,
    val rating: ProviderRating,
    val deliveryAvailable: Boolean,
    val deliveryCharge: Money?
)

enum class EquipmentType {
    TRACTOR, HARVESTER, SPRAYER, THRESHER, PLOUGH, SEED_DRILL
}

data class EquipmentSpecs(
    val horsepower: Int?,
    val capacity: String?,
    val age: Int?,
    val fuelType: String?
)

data class Transaction(
    val id: String,
    val userId: String,
    val type: TransactionType,
    val providerId: String,
    val providerName: String,
    val details: TransactionDetails,
    val status: TransactionStatus,
    val createdAt: Instant,
    val updatedAt: Instant,
    val completedAt: Instant?
)

enum class TransactionType {
    CROP_SALE, COLD_STORAGE_BOOKING, EQUIPMENT_RENTAL
}

sealed class TransactionDetails {
    data class CropSale(
        val cropType: CropType,
        val quantity: Quantity,
        val pricePerUnit: Money,
        val totalAmount: Money,
        val pickupDetails: PickupDetails
    ) : TransactionDetails()
    
    data class ColdStorage(
        val duration: Int,
        val capacity: Capacity,
        val dailyRate: Money,
        val totalAmount: Money,
        val startDate: Instant,
        val endDate: Instant
    ) : TransactionDetails()
    
    data class Equipment(
        val equipmentType: EquipmentType,
        val startDate: Instant,
        val endDate: Instant,
        val dailyRate: Money,
        val totalAmount: Money,
        val deliveryIncluded: Boolean
    ) : TransactionDetails()
}

enum class TransactionStatus {
    INITIATED, CONFIRMED, IN_PROGRESS, COMPLETED, CANCELLED, FAILED
}

data class Quantity(
    val value: Double,
    val unit: QuantityUnit
)

enum class QuantityUnit {
    KILOGRAM, QUINTAL, TON
}
```

#### User Profile

```kotlin
data class UserProfile(
    val userId: String,
    val phoneNumber: String,
    val languageCode: String,
    val defaultLocation: Location,
    val primaryCrops: List<CropType>,
    val createdAt: Instant,
    val lastSyncedAt: Instant?
)

data class Location(
    val village: String?,
    val district: String,
    val state: String,
    val coordinates: GeoLocation?
)

enum class CropType {
    RICE, WHEAT, COTTON, TOMATO, POTATO, SUGARCANE, MAIZE, SOYBEAN, ONION, CHILI
}
```


## Security Design

### Encryption Strategies

#### Data at Rest

**Android Local Storage**:
```kotlin
class EncryptedStorageManager(private val context: Context) {
    
    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()
    
    // Encrypted SharedPreferences for sensitive data
    private val encryptedPrefs = EncryptedSharedPreferences.create(
        context,
        "secure_prefs",
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )
    
    // Room database encryption
    private val database = Room.databaseBuilder(
        context,
        AgriEdgeDatabase::class.java,
        "agriedge.db"
    )
    .openHelperFactory(SupportFactory(getDatabasePassphrase()))
    .build()
    
    private fun getDatabasePassphrase(): ByteArray {
        // Generate or retrieve passphrase from Android Keystore
        val keyStore = KeyStore.getInstance("AndroidKeyStore")
        keyStore.load(null)
        
        if (!keyStore.containsAlias(DB_KEY_ALIAS)) {
            generateDatabaseKey()
        }
        
        return retrieveDatabaseKey()
    }
    
    private fun generateDatabaseKey() {
        val keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            "AndroidKeyStore"
        )
        
        val keyGenParameterSpec = KeyGenParameterSpec.Builder(
            DB_KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
        .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
        .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
        .setKeySize(256)
        .build()
        
        keyGenerator.init(keyGenParameterSpec)
        keyGenerator.generateKey()
    }
    
    companion object {
        private const val DB_KEY_ALIAS = "agriedge_db_key"
    }
}
```

**Backend Database**:
- PostgreSQL Transparent Data Encryption (TDE) enabled
- Encrypted backups using AWS KMS or GCP Cloud KMS
- Column-level encryption for sensitive fields (phone numbers)

```kotlin
// Column-level encryption for sensitive data
@Entity
data class UserEntity(
    @Id
    val userId: UUID,
    
    @Convert(converter = EncryptedStringConverter::class)
    val phoneNumber: String,
    
    val languageCode: String,
    // ... other fields
)

class EncryptedStringConverter : AttributeConverter<String, String> {
    
    private val cipher = Cipher.getInstance("AES/GCM/NoPadding")
    private val secretKey = loadEncryptionKey()
    
    override fun convertToDatabaseColumn(attribute: String?): String? {
        if (attribute == null) return null
        
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)
        val iv = cipher.iv
        val encrypted = cipher.doFinal(attribute.toByteArray())
        
        return Base64.getEncoder().encodeToString(iv + encrypted)
    }
    
    override fun convertToEntityAttribute(dbData: String?): String? {
        if (dbData == null) return null
        
        val decoded = Base64.getDecoder().decode(dbData)
        val iv = decoded.sliceArray(0..11)
        val encrypted = decoded.sliceArray(12 until decoded.size)
        
        cipher.init(Cipher.DECRYPT_MODE, secretKey, GCMParameterSpec(128, iv))
        return String(cipher.doFinal(encrypted))
    }
    
    private fun loadEncryptionKey(): SecretKey {
        // Load from environment variable or secrets manager
        val keyBytes = System.getenv("DB_ENCRYPTION_KEY").decodeBase64()
        return SecretKeySpec(keyBytes, "AES")
    }
}
```

#### Data in Transit

**TLS Configuration**:
```kotlin
// Android OkHttp client
class NetworkModule {
    
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectionSpecs(listOf(
                ConnectionSpec.Builder(ConnectionSpec.MODERN_TLS)
                    .tlsVersions(TlsVersion.TLS_1_3, TlsVersion.TLS_1_2)
                    .cipherSuites(
                        CipherSuite.TLS_AES_128_GCM_SHA256,
                        CipherSuite.TLS_AES_256_GCM_SHA384,
                        CipherSuite.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                        CipherSuite.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
                    )
                    .build()
            ))
            .certificatePinner(
                CertificatePinner.Builder()
                    .add("api.agriedge.com", "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
                    .build()
            )
            .addInterceptor(AuthInterceptor())
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = if (BuildConfig.DEBUG) {
                    HttpLoggingInterceptor.Level.BODY
                } else {
                    HttpLoggingInterceptor.Level.NONE
                }
            })
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }
}
```

**Backend TLS Configuration**:
```yaml
# application.yml
server:
  ssl:
    enabled: true
    protocol: TLS
    enabled-protocols: TLSv1.3,TLSv1.2
    ciphers: TLS_AES_128_GCM_SHA256,TLS_AES_256_GCM_SHA384,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
    key-store: classpath:keystore.p12
    key-store-password: ${SSL_KEYSTORE_PASSWORD}
    key-store-type: PKCS12
```

### API Security

#### Rate Limiting

```kotlin
@Configuration
class RateLimitConfig {
    
    @Bean
    fun rateLimiter(): RateLimiter {
        return RateLimiter.create(
            RateLimiterConfig.custom()
                .limitForPeriod(100)  // 100 requests
                .limitRefreshPeriod(Duration.ofMinutes(1))  // per minute
                .timeoutDuration(Duration.ofSeconds(5))
                .build()
        )
    }
}

@Component
class RateLimitInterceptor(
    private val rateLimiter: RateLimiter
) : HandlerInterceptor {
    
    override fun preHandle(
        request: HttpServletRequest,
        response: HttpServletResponse,
        handler: Any
    ): Boolean {
        val userId = extractUserId(request)
        val userRateLimiter = getUserRateLimiter(userId)
        
        return if (userRateLimiter.acquirePermission()) {
            true
        } else {
            response.status = 429
            response.writer.write("""{"error": "Rate limit exceeded"}""")
            false
        }
    }
}
```

#### Input Validation

```kotlin
// Request DTOs with validation
data class DiagnosisSyncRequest(
    @field:Valid
    @field:Size(min = 1, max = 100, message = "Batch size must be between 1 and 100")
    val diagnoses: List<DiagnosisDto>
)

data class DiagnosisDto(
    @field:NotBlank
    @field:Pattern(regexp = UUID_REGEX)
    val id: String,
    
    @field:NotNull
    @field:Min(0)
    val timestamp: Long,
    
    @field:NotBlank
    val cropType: String,
    
    @field:NotBlank
    val diseaseId: String,
    
    @field:DecimalMin("0.0")
    @field:DecimalMax("1.0")
    val confidence: Float,
    
    @field:Pattern(regexp = URL_REGEX)
    val imageUrl: String?,
    
    @field:DecimalMin("-90.0")
    @field:DecimalMax("90.0")
    val latitude: Double?,
    
    @field:DecimalMin("-180.0")
    @field:DecimalMax("180.0")
    val longitude: Double?
)
```

#### SQL Injection Prevention

```kotlin
// Using parameterized queries with JPA
@Repository
interface DiagnosisRepository : JpaRepository<DiagnosisEntity, UUID> {
    
    @Query("SELECT d FROM DiagnosisEntity d WHERE d.userId = :userId AND d.timestamp >= :startTime")
    fun findByUserIdAndTimestamp(
        @Param("userId") userId: UUID,
        @Param("startTime") startTime: Instant
    ): List<DiagnosisEntity>
}
```

### Data Privacy Measures

#### Anonymization

```kotlin
class TelemetryAnonymizer {
    
    fun anonymize(event: TelemetryEventDto): AnonymizedTelemetryEvent {
        return AnonymizedTelemetryEvent(
            eventType = event.eventType,
            timestamp = event.timestamp,
            deviceModel = event.deviceModel,
            osVersion = event.osVersion,
            appVersion = event.appVersion,
            metadata = anonymizeMetadata(event.metadata),
            // User ID is hashed, not stored directly
            userHash = hashUserId(event.userId)
        )
    }
    
    private fun anonymizeMetadata(metadata: Map<String, Any>): Map<String, Any> {
        return metadata.filterKeys { key ->
            // Remove any potentially identifying information
            key !in listOf("phoneNumber", "location", "name", "address")
        }
    }
    
    private fun hashUserId(userId: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(userId.toByteArray())
        return Base64.getEncoder().encodeToString(hash)
    }
}
```

#### GDPR Compliance

```kotlin
@Service
class DataDeletionService(
    private val userRepository: UserRepository,
    private val diagnosisRepository: DiagnosisRepository,
    private val transactionRepository: TransactionRepository,
    private val s3Client: S3Client
) {
    
    suspend fun deleteUserData(userId: UUID) {
        // 1. Delete user profile
        userRepository.deleteById(userId)
        
        // 2. Delete diagnoses
        val diagnoses = diagnosisRepository.findByUserId(userId)
        diagnoses.forEach { diagnosis ->
            // Delete images from S3
            diagnosis.imageUrl?.let { url ->
                val key = extractS3Key(url)
                s3Client.deleteObject { it.bucket("agriedge-images").key(key) }
            }
        }
        diagnosisRepository.deleteByUserId(userId)
        
        // 3. Anonymize transactions (keep for audit)
        transactionRepository.anonymizeUserTransactions(userId)
        
        // 4. Delete telemetry
        telemetryRepository.deleteByUserId(userId)
        
        // 5. Revoke tokens
        refreshTokenRepository.revokeAllForUser(userId)
    }
}
```

## Performance Optimization

### ML Model Quantization

#### Quantization Strategy

```python
# Model quantization script (Python)
import tensorflow as tf

def quantize_model(model_path, output_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use integer quantization with float fallback
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Representative dataset for calibration
    def representative_dataset():
        for _ in range(100):
            # Generate random images similar to training data
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    
    # Enable full integer quantization
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

# Validate accuracy after quantization
def validate_quantized_model(tflite_path, test_dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    
    for image, label in test_dataset:
        # Preprocess
        input_data = preprocess_image(image)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Check prediction
        predicted = np.argmax(output_data)
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"Quantized model accuracy: {accuracy:.4f}")
    return accuracy
```

### Image Compression

```kotlin
class ImageCompressor {
    
    fun compressForUpload(bitmap: Bitmap): ByteArray {
        val outputStream = ByteArrayOutputStream()
        
        // Resize if too large
        val resized = if (bitmap.width > MAX_DIMENSION || bitmap.height > MAX_DIMENSION) {
            val scale = MAX_DIMENSION.toFloat() / maxOf(bitmap.width, bitmap.height)
            Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).toInt(),
                (bitmap.height * scale).toInt(),
                true
            )
        } else {
            bitmap
        }
        
        // Compress as JPEG
        resized.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
        
        return outputStream.toByteArray()
    }
    
    fun generateThumbnail(bitmap: Bitmap): ByteArray {
        val thumbnail = Bitmap.createScaledBitmap(
            bitmap,
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
            true
        )
        
        val outputStream = ByteArrayOutputStream()
        thumbnail.compress(Bitmap.CompressFormat.JPEG, THUMBNAIL_QUALITY, outputStream)
        
        return outputStream.toByteArray()
    }
    
    companion object {
        private const val MAX_DIMENSION = 1024
        private const val JPEG_QUALITY = 85
        private const val THUMBNAIL_SIZE = 256
        private const val THUMBNAIL_QUALITY = 75
    }
}
```

### Caching Strategies

#### Android Caching

```kotlin
class CacheManager(
    private val context: Context,
    private val database: AgriEdgeDatabase
) {
    
    // In-memory cache for frequently accessed data
    private val treatmentCache = LruCache<String, Treatment>(50)
    private val diseaseCache = LruCache<String, Disease>(100)
    
    // Disk cache for images
    private val imageCache = DiskLruCache.open(
        File(context.cacheDir, "images"),
        1,
        1,
        50 * 1024 * 1024  // 50 MB
    )
    
    suspend fun getTreatment(diseaseId: String, cropType: CropType): Treatment? {
        // Check memory cache
        val cacheKey = "$diseaseId-$cropType"
        treatmentCache.get(cacheKey)?.let { return it }
        
        // Check database
        val treatment = database.treatmentDao().getTreatment(diseaseId, cropType.name)
        treatment?.let {
            treatmentCache.put(cacheKey, it.toDomain())
            return it.toDomain()
        }
        
        return null
    }
    
    suspend fun cacheImage(url: String, bitmap: Bitmap) {
        withContext(Dispatchers.IO) {
            val editor = imageCache.edit(url.md5())
            editor?.let {
                val outputStream = it.newOutputStream(0)
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                outputStream.close()
                it.commit()
            }
        }
    }
}
```

#### Backend Caching

```kotlin
@Configuration
@EnableCaching
class CacheConfig {
    
    @Bean
    fun cacheManager(redisConnectionFactory: RedisConnectionFactory): CacheManager {
        val config = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))
            .serializeKeysWith(
                RedisSerializationContext.SerializationPair.fromSerializer(
                    StringRedisSerializer()
                )
            )
            .serializeValuesWith(
                RedisSerializationContext.SerializationPair.fromSerializer(
                    GenericJackson2JsonRedisSerializer()
                )
            )
        
        return RedisCacheManager.builder(redisConnectionFactory)
            .cacheDefaults(config)
            .withCacheConfiguration("treatments", config.entryTtl(Duration.ofDays(7)))
            .withCacheConfiguration("providers", config.entryTtl(Duration.ofMinutes(30)))
            .build()
    }
}

@Service
class TreatmentService(
    private val treatmentRepository: TreatmentRepository
) {
    
    @Cacheable(value = ["treatments"], key = "#diseaseId + '-' + #cropType")
    suspend fun getTreatments(diseaseId: String, cropType: String): List<Treatment> {
        return treatmentRepository.findByDiseaseIdAndCropType(diseaseId, cropType)
    }
}
```

### Background Sync Optimization

```kotlin
class OptimizedSyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        // Check battery level
        if (isBatteryLow()) {
            return Result.retry()
        }
        
        // Check network type
        val networkType = getNetworkType()
        if (networkType == NetworkType.METERED && !userAllowsMeteredSync()) {
            return Result.retry()
        }
        
        // Batch sync items
        val batchSize = when (networkType) {
            NetworkType.WIFI -> 50
            NetworkType.METERED -> 10
            else -> 5
        }
        
        val pendingItems = syncQueueDao.getPendingItems(limit = batchSize)
        
        // Sync in parallel with limited concurrency
        val results = pendingItems.chunked(5).flatMap { chunk ->
            chunk.map { item ->
                async(Dispatchers.IO) {
                    syncItem(item)
                }
            }.awaitAll()
        }
        
        val allSucceeded = results.all { it }
        
        return if (allSucceeded) {
            Result.success()
        } else {
            Result.retry()
        }
    }
    
    private fun isBatteryLow(): Boolean {
        val batteryManager = applicationContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val batteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        return batteryLevel < 20
    }
}
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies and consolidations:

**Redundancy Analysis**:
1. Properties 17.1, 21.1, 24.1 (displaying all required fields for different provider types) can be consolidated into a single property about provider result completeness
2. Properties 5.2 and 18.2 (displaying all required fields for diagnoses and transactions) are similar patterns about data completeness
3. Properties 19.1, 19.2, 22.3, 25.2, 25.3 (displaying confirmation details) follow the same pattern
4. Properties 29.1, 29.2, 29.3 (provider rating display) can be consolidated into one comprehensive property
5. Properties 43.1 and 43.2 (encryption of different data types) can be combined into one property about encryption coverage
6. Properties 44.1, 44.2, 44.3 (TLS requirements) can be consolidated into one comprehensive security property

**Consolidated Properties**:
- Provider result completeness (covers buyers, cold storage, equipment)
- Data completeness for stored entities (covers diagnoses, transactions)
- Confirmation details completeness (covers all transaction types)
- Provider rating display completeness
- Encryption coverage for all sensitive data
- Network security requirements

### Core Diagnostic Properties

**Property 1: Image Quality Validation**
*For any* captured image, the validation function should correctly identify quality issues (blur, poor lighting, insufficient resolution, missing leaf content) and reject images that fail any quality check.
**Validates: Requirements 1.3**

**Property 2: Inference Performance**
*For any* valid input image, the diagnostic engine should complete classification and return results within 3 seconds on target hardware (Snapdragon 665 or equivalent).
**Validates: Requirements 2.2**

**Property 3: Confidence Score Display**
*For any* classification result, the system should display the confidence score as a percentage and show exactly 3 disease predictions with their respective confidence scores.
**Validates: Requirements 3.1, 3.3**

**Property 4: Treatment Completeness**
*For any* identified disease, the system should provide treatment recommendations that include both organic and chemical options, application timing, and dosage information.
**Validates: Requirements 4.1, 4.2, 4.3**

**Property 5: Diagnosis Persistence**
*For any* completed diagnosis, the system should store it in local storage with all required fields (timestamp, crop type, disease name, confidence score, and photograph reference).
**Validates: Requirements 5.1, 5.2**

**Property 6: Diagnosis Retention**
*For any* diagnosis stored within the last 12 months, it should remain accessible in local storage and not be automatically deleted.
**Validates: Requirements 5.3**

**Property 7: Pest Recommendation**
*For any* pest detection result, the system should provide pest-specific control recommendations.
**Validates: Requirements 7.2**

**Property 8: History Sorting**
*For any* set of diagnoses in history, they should be displayed in reverse chronological order (most recent first).
**Validates: Requirements 8.1**

**Property 9: History Display Completeness**
*For any* diagnosis displayed in history, it should include thumbnail image, disease name, date, and confidence score.
**Validates: Requirements 8.2**

**Property 10: History Query Performance**
*For any* diagnostic history query from local storage, results should be returned within 500 milliseconds.
**Validates: Requirements 8.3**

### Voice Interface Properties

**Property 11: Command Parsing**
*For any* valid voice command (diagnostic, market search, or history), the voice interface should correctly parse the command type and extract relevant parameters (crop type, quantity, intent).
**Validates: Requirements 11.1, 11.2, 11.4**

**Property 12: Offline Functionality**
*For any* core function (diagnosis, history viewing, market search), it should remain operational when voice services are unavailable, using text-only mode.
**Validates: Requirements 12.3**

### Localization Properties

**Property 13: UI Localization**
*For any* UI element, the displayed text should be in the user's selected language.
**Validates: Requirements 13.1**

**Property 14: Language Persistence**
*For any* language preference setting, it should persist across app restarts and be restored when the app is relaunched.
**Validates: Requirements 13.3**

**Property 15: Disease Name Localization**
*For any* disease display, both the common name and scientific name should be shown in the user's selected language.
**Validates: Requirements 14.1**

**Property 16: Treatment Alternatives**
*For any* treatment recommendation, the system should provide alternative treatment options in addition to the primary recommendation.
**Validates: Requirements 15.3**

### Market Properties

**Property 17: Search Performance**
*For any* market search request (buyers, cold storage, or equipment), results should be returned within 5 seconds.
**Validates: Requirements 16.2**

**Property 18: Provider Result Completeness**
*For any* provider result (buyer, cold storage facility, or equipment rental), all required fields should be present: provider name, pricing information, distance, rating, and type-specific details (pickup details for buyers, capacity for cold storage, specifications for equipment).
**Validates: Requirements 17.1, 21.1, 21.2, 21.3, 24.1, 24.2, 24.3**

**Property 19: Price Calculation**
*For any* buyer quote and specified quantity, the displayed total payment amount should equal price per unit multiplied by quantity.
**Validates: Requirements 17.2**

**Property 20: Result Sorting**
*For any* set of buyer results, they should be sorted by price (highest to lowest) by default. For cold storage results, they should be sorted by distance (nearest first).
**Validates: Requirements 17.3, 20.3**

**Property 21: Transaction Completeness**
*For any* initiated or confirmed transaction, all required details should be present: transaction ID, crop type (if applicable), quantity, price, and pickup/delivery details.
**Validates: Requirements 18.2, 18.3, 19.1, 19.2, 22.3, 25.2, 25.3**

**Property 22: Reminder Scheduling**
*For any* confirmed transaction with a scheduled pickup time, the system should schedule reminders at 24 hours and 2 hours before the pickup time.
**Validates: Requirements 19.3**

**Property 23: Equipment Availability Display**
*For any* equipment search result, availability status for the requested dates should be displayed.
**Validates: Requirements 23.3**

**Property 24: Logistics Bundling Indication**
*For any* buyer quote, the system should clearly indicate whether logistics bundling is available.
**Validates: Requirements 26.2**

**Property 25: Transport Option Completeness**
*For any* transport option in a bundled quote, vehicle type, capacity, cost, and estimated timing should be displayed.
**Validates: Requirements 27.2, 27.3**

**Property 26: Bundled Price Calculation**
*For any* bundled transaction quote, the displayed combined price should equal the sum of the buyer price and transport cost.
**Validates: Requirements 28.2**

**Property 27: Bundled Transaction ID**
*For any* bundled transaction (sale + transport), a single transaction ID should cover both components.
**Validates: Requirements 28.3**

**Property 28: Provider Rating Display**
*For any* provider result, the system should display average rating (1-5 stars), total number of completed transactions, and rating distribution (counts for 5-star, 4-star, 3-star, 2-star, 1-star).
**Validates: Requirements 29.1, 29.2, 29.3**

**Property 29: Review Completeness**
*For any* displayed review, it should include review text, rating, and date.
**Validates: Requirements 30.1, 30.2**

**Property 30: Rating Sync**
*For any* submitted rating, it should be queued for synchronization and uploaded when connectivity is available.
**Validates: Requirements 31.3**

### Authentication Properties

**Property 31: Phone Number Validation**
*For any* phone number input, the validation function should correctly accept valid 10-digit Indian phone numbers and reject invalid formats.
**Validates: Requirements 32.1**

**Property 32: Profile Persistence**
*For any* created or updated user profile, it should be saved to local storage and retrievable on subsequent app launches.
**Validates: Requirements 34.3**

**Property 33: Profile Cloud Sync**
*For any* user profile when cloud backup is enabled, profile data should be queued for synchronization to cloud storage.
**Validates: Requirements 34.4**

**Property 34: Authentication Persistence**
*For any* authenticated session, the authentication state should persist across app restarts until the user explicitly logs out.
**Validates: Requirements 35.2**

### Sync Properties

**Property 35: Offline Event Queuing**
*For any* diagnostic event that occurs while offline, it should be added to the sync queue with timestamp and event type.
**Validates: Requirements 36.1, 36.2**

**Property 36: Sync Queue Crash Resilience**
*For any* queued sync item, it should remain in the queue even if the application crashes before sync completes.
**Validates: Requirements 36.3**

**Property 37: Sync Timing**
*For any* set of queued events when network connectivity is restored, all events should be uploaded within 5 minutes.
**Validates: Requirements 37.1**

**Property 38: Sync Prioritization**
*For any* sync operation with multiple event types queued, diagnostic events should be synced before other data types.
**Validates: Requirements 37.3**

**Property 39: Sync Progress Accuracy**
*For any* ongoing sync operation, the displayed progress (number of pending items and upload progress) should accurately reflect the actual sync state.
**Validates: Requirements 38.2**

**Property 40: Sync Retry Logic**
*For any* failed sync attempt, the system should retry with exponential backoff (30s, 2m, 10m) up to 3 attempts before marking as failed.
**Validates: Requirements 39.1**

**Property 41: Sync Network Recovery**
*For any* sync item that failed due to network issues, the system should automatically retry when network conditions improve.
**Validates: Requirements 39.3**

### Performance Properties

**Property 42: Detail Load Performance**
*For any* historical diagnosis detail view, the data should be loaded and displayed within 300 milliseconds.
**Validates: Requirements 41.2**

### Security Properties

**Property 43: Data Encryption Coverage**
*For any* locally stored sensitive data (diagnostic images, user profile data, transaction history), it should be encrypted using AES-256 encryption.
**Validates: Requirements 43.1, 43.2**

**Property 44: Network Security**
*For any* network request to remote APIs, the connection should use TLS 1.3 or higher, validate SSL certificates, and refuse to transmit data over insecure connections.
**Validates: Requirements 44.1, 44.2, 44.3**

**Property 45: Telemetry Anonymization**
*For any* telemetry data before transmission, all personally identifiable information should be removed, and only device type, OS version, and diagnostic accuracy metrics should be included.
**Validates: Requirements 45.1, 45.2, 45.3**

### Data Model Invariants

**Property 46: Diagnosis Data Integrity**
*For any* diagnosis entity, the confidence score should be between 0.0 and 1.0, the timestamp should not be in the future, and the crop type should be one of the supported types.

**Property 47: Transaction Amount Consistency**
*For any* transaction entity, if price per unit and quantity are present, the total amount should equal their product.

**Property 48: Sync Queue Ordering**
*For any* set of items in the sync queue, they should be processed in FIFO order (first in, first out) based on timestamp.

**Property 49: Rating Value Range**
*For any* provider rating, the rating value should be an integer between 1 and 5 (inclusive).

**Property 50: Location Coordinate Validity**
*For any* location with coordinates, latitude should be between -90 and 90, and longitude should be between -180 and 180.


## Error Handling

### Error Categories

#### Network Errors
- **Connection Timeout**: Retry with exponential backoff
- **No Internet**: Queue for later sync, continue offline
- **Server Error (5xx)**: Retry up to 3 times
- **Client Error (4xx)**: Display error to user, don't retry

#### ML Model Errors
- **Model Not Found**: Download model from CDN
- **Model Corrupted**: Re-download model, use cached version as fallback
- **Inference Failure**: Log error, prompt user to retry
- **Out of Memory**: Reduce image resolution, retry inference

#### Data Errors
- **Database Corruption**: Attempt recovery, restore from backup if available
- **Disk Full**: Prompt user to clear space, offer to delete old diagnoses
- **Encryption Failure**: Log error, prevent data write

#### External API Errors
- **Bhashini API Unavailable**: Fall back to text-only mode
- **Beckn Network Timeout**: Display cached results if available
- **SMS Gateway Failure**: Allow manual OTP entry, provide support contact

### Error Handling Strategy

```kotlin
sealed class AppError {
    data class NetworkError(val type: NetworkErrorType, val message: String) : AppError()
    data class ModelError(val type: ModelErrorType, val message: String) : AppError()
    data class DataError(val type: DataErrorType, val message: String) : AppError()
    data class ValidationError(val field: String, val message: String) : AppError()
    data class UnknownError(val throwable: Throwable) : AppError()
}

enum class NetworkErrorType {
    TIMEOUT, NO_INTERNET, SERVER_ERROR, CLIENT_ERROR, SSL_ERROR
}

enum class ModelErrorType {
    NOT_FOUND, CORRUPTED, INFERENCE_FAILED, OUT_OF_MEMORY
}

enum class DataErrorType {
    DATABASE_CORRUPTED, DISK_FULL, ENCRYPTION_FAILED, SYNC_FAILED
}

class ErrorHandler {
    
    fun handle(error: AppError): ErrorAction {
        return when (error) {
            is AppError.NetworkError -> handleNetworkError(error)
            is AppError.ModelError -> handleModelError(error)
            is AppError.DataError -> handleDataError(error)
            is AppError.ValidationError -> ErrorAction.ShowMessage(error.message)
            is AppError.UnknownError -> handleUnknownError(error)
        }
    }
    
    private fun handleNetworkError(error: AppError.NetworkError): ErrorAction {
        return when (error.type) {
            NetworkErrorType.TIMEOUT -> ErrorAction.Retry(maxAttempts = 3)
            NetworkErrorType.NO_INTERNET -> ErrorAction.QueueForLater
            NetworkErrorType.SERVER_ERROR -> ErrorAction.Retry(maxAttempts = 3)
            NetworkErrorType.CLIENT_ERROR -> ErrorAction.ShowMessage(error.message)
            NetworkErrorType.SSL_ERROR -> ErrorAction.ShowMessage("Secure connection failed")
        }
    }
    
    private fun handleModelError(error: AppError.ModelError): ErrorAction {
        return when (error.type) {
            ModelErrorType.NOT_FOUND -> ErrorAction.DownloadModel
            ModelErrorType.CORRUPTED -> ErrorAction.RedownloadModel
            ModelErrorType.INFERENCE_FAILED -> ErrorAction.ShowMessage("Diagnosis failed. Please try again.")
            ModelErrorType.OUT_OF_MEMORY -> ErrorAction.ReduceImageQuality
        }
    }
    
    private fun handleDataError(error: AppError.DataError): ErrorAction {
        return when (error.type) {
            DataErrorType.DATABASE_CORRUPTED -> ErrorAction.RecoverDatabase
            DataErrorType.DISK_FULL -> ErrorAction.PromptClearSpace
            DataErrorType.ENCRYPTION_FAILED -> ErrorAction.LogAndAbort
            DataErrorType.SYNC_FAILED -> ErrorAction.QueueForLater
        }
    }
}

sealed class ErrorAction {
    data class Retry(val maxAttempts: Int) : ErrorAction()
    object QueueForLater : ErrorAction()
    data class ShowMessage(val message: String) : ErrorAction()
    object DownloadModel : ErrorAction()
    object RedownloadModel : ErrorAction()
    object ReduceImageQuality : ErrorAction()
    object RecoverDatabase : ErrorAction()
    object PromptClearSpace : ErrorAction()
    object LogAndAbort : ErrorAction()
}
```

### Graceful Degradation

```kotlin
class FeatureManager {
    
    fun getAvailableFeatures(): Set<Feature> {
        val features = mutableSetOf<Feature>()
        
        // Core features always available
        features.add(Feature.DIAGNOSIS)
        features.add(Feature.HISTORY)
        
        // Network-dependent features
        if (isNetworkAvailable()) {
            features.add(Feature.MARKET_SEARCH)
            features.add(Feature.SYNC)
        }
        
        // Voice features
        if (isBhashiniAvailable()) {
            features.add(Feature.VOICE_INPUT)
            features.add(Feature.VOICE_OUTPUT)
        }
        
        // GPU acceleration
        if (isGpuAvailable()) {
            features.add(Feature.GPU_ACCELERATION)
        }
        
        return features
    }
    
    enum class Feature {
        DIAGNOSIS,
        HISTORY,
        MARKET_SEARCH,
        SYNC,
        VOICE_INPUT,
        VOICE_OUTPUT,
        GPU_ACCELERATION
    }
}
```

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests as complementary approaches:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Verify concrete scenarios with known inputs and outputs
- Test error conditions and boundary cases
- Validate integration between components
- Fast execution for rapid feedback

**Property-Based Tests**: Verify universal properties across all inputs
- Generate hundreds of random inputs to test properties
- Discover edge cases that manual testing might miss
- Validate invariants and business rules
- Provide high confidence in correctness

Both approaches are necessary for comprehensive coverage. Unit tests catch specific bugs and validate examples, while property tests ensure general correctness across the input space.

### Property-Based Testing Configuration

**Framework Selection**:
- **Android/Kotlin**: Kotest Property Testing
- **Backend/Kotlin**: Kotest Property Testing
- **Python (ML)**: Hypothesis

**Test Configuration**:
```kotlin
// Kotest configuration
class DiagnosisPropertyTest : StringSpec({
    
    "Property 1: Image Quality Validation" {
        checkAll(
            iterations = 100,
            Arb.bitmap()
        ) { bitmap ->
            val result = imageQualityValidator.validate(bitmap)
            
            // Property: validation should identify all quality issues
            if (result is ValidationResult.Invalid) {
                result.reasons.shouldNotBeEmpty()
                result.reasons.forEach { reason ->
                    reason.shouldBeOneOf(
                        "Image too dark",
                        "Image too bright",
                        "Image too blurry",
                        "Resolution too low",
                        "No leaf detected"
                    )
                }
            }
        }
    }
    
    "Property 5: Diagnosis Persistence" {
        checkAll(
            iterations = 100,
            Arb.diagnosis()
        ) { diagnosis ->
            // Save diagnosis
            diagnosisRepository.save(diagnosis)
            
            // Retrieve diagnosis
            val retrieved = diagnosisRepository.getById(diagnosis.id)
            
            // Property: all required fields should be present
            retrieved.shouldNotBeNull()
            retrieved.id shouldBe diagnosis.id
            retrieved.timestamp shouldBe diagnosis.timestamp
            retrieved.cropType shouldBe diagnosis.cropType
            retrieved.diseaseName shouldBe diagnosis.diseaseName
            retrieved.confidence shouldBe diagnosis.confidence
            retrieved.imagePath.shouldNotBeNull()
        }
    }
})

// Custom generators
fun Arb.Companion.diagnosis(): Arb<Diagnosis> = arbitrary {
    Diagnosis(
        id = UUID.randomUUID().toString(),
        userId = UUID.randomUUID().toString(),
        timestamp = Instant.now(),
        cropType = Arb.enum<CropType>().bind(),
        disease = Arb.disease().bind(),
        confidence = Arb.float(0.0f, 1.0f).bind(),
        image = DiagnosisImage(
            localPath = "/path/to/image.jpg",
            remoteUrl = null,
            thumbnailUrl = null
        ),
        location = Arb.geoLocation().orNull().bind(),
        synced = false
    )
}

fun Arb.Companion.geoLocation(): Arb<GeoLocation> = arbitrary {
    GeoLocation(
        latitude = Arb.double(-90.0, 90.0).bind(),
        longitude = Arb.double(-180.0, 180.0).bind()
    )
}
```

**Test Tagging**:
Each property test must include a comment referencing the design document property:

```kotlin
/**
 * Feature: agriedge-link, Property 8: History Sorting
 * 
 * For any set of diagnoses in history, they should be displayed 
 * in reverse chronological order (most recent first).
 */
@Test
fun `property test - history sorting`() {
    checkAll(
        iterations = 100,
        Arb.list(Arb.diagnosis(), 1..50)
    ) { diagnoses ->
        val sorted = diagnosisRepository.getHistory(userId)
        
        // Property: should be sorted by timestamp descending
        sorted.zipWithNext().forEach { (current, next) ->
            current.timestamp shouldBeGreaterThanOrEqualTo next.timestamp
        }
    }
}
```

### Unit Testing Strategy

**Test Organization**:
```
test/
├── unit/
│   ├── domain/
│   │   ├── DiagnosisUseCaseTest.kt
│   │   ├── MarketSearchUseCaseTest.kt
│   │   └── SyncUseCaseTest.kt
│   ├── data/
│   │   ├── DiagnosisRepositoryTest.kt
│   │   ├── UserProfileRepositoryTest.kt
│   │   └── SyncQueueRepositoryTest.kt
│   └── ml/
│       ├── DiseaseClassifierTest.kt
│       └── ImagePreprocessorTest.kt
├── integration/
│   ├── DatabaseIntegrationTest.kt
│   ├── NetworkIntegrationTest.kt
│   └── BecknIntegrationTest.kt
└── property/
    ├── DiagnosisPropertyTest.kt
    ├── MarketPropertyTest.kt
    ├── SyncPropertyTest.kt
    └── SecurityPropertyTest.kt
```

**Example Unit Tests**:
```kotlin
class DiagnosisUseCaseTest : StringSpec({
    
    val mockRepository = mockk<DiagnosisRepository>()
    val mockClassifier = mockk<DiseaseClassifier>()
    val useCase = DiagnoseDiseaseUseCase(mockRepository, mockClassifier)
    
    "should classify disease and save diagnosis" {
        // Arrange
        val bitmap = createTestBitmap()
        val cropType = CropType.COTTON
        val classificationResult = ClassificationResult(
            topPredictions = listOf(
                Prediction("cotton_leaf_curl", "Cotton Leaf Curl", 0.92f, CropType.COTTON)
            ),
            inferenceTime = 2500L
        )
        
        every { mockClassifier.classify(bitmap) } returns classificationResult
        coEvery { mockRepository.save(any()) } just Runs
        
        // Act
        val result = useCase.execute(bitmap, cropType)
        
        // Assert
        result.shouldBeInstanceOf<Result.Success<Diagnosis>>()
        val diagnosis = result.data
        diagnosis.disease.name shouldBe "Cotton Leaf Curl"
        diagnosis.confidence shouldBe 0.92f
        
        coVerify { mockRepository.save(any()) }
    }
    
    "should handle low confidence scores" {
        // Arrange
        val bitmap = createTestBitmap()
        val cropType = CropType.COTTON
        val classificationResult = ClassificationResult(
            topPredictions = listOf(
                Prediction("cotton_leaf_curl", "Cotton Leaf Curl", 0.65f, CropType.COTTON)
            ),
            inferenceTime = 2500L
        )
        
        every { mockClassifier.classify(bitmap) } returns classificationResult
        
        // Act
        val result = useCase.execute(bitmap, cropType)
        
        // Assert
        result.shouldBeInstanceOf<Result.Success<Diagnosis>>()
        result.data.requiresExpertConsultation shouldBe true
    }
})
```

### Integration Testing

```kotlin
@RunWith(AndroidJUnit4::class)
class DatabaseIntegrationTest {
    
    private lateinit var database: AgriEdgeDatabase
    private lateinit var diagnosisDao: DiagnosisDao
    
    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        database = Room.inMemoryDatabaseBuilder(context, AgriEdgeDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        diagnosisDao = database.diagnosisDao()
    }
    
    @After
    fun teardown() {
        database.close()
    }
    
    @Test
    fun testDiagnosisCRUD() = runBlocking {
        // Create
        val diagnosis = createTestDiagnosis()
        diagnosisDao.insertDiagnosis(diagnosis)
        
        // Read
        val retrieved = diagnosisDao.getDiagnosisById(diagnosis.id)
        assertNotNull(retrieved)
        assertEquals(diagnosis.id, retrieved?.id)
        
        // Update
        val updated = diagnosis.copy(synced = true)
        diagnosisDao.markAsSynced(diagnosis.id)
        val afterUpdate = diagnosisDao.getDiagnosisById(diagnosis.id)
        assertTrue(afterUpdate?.synced == true)
        
        // Delete
        diagnosisDao.deleteOldDiagnoses(System.currentTimeMillis() + 1000)
        val afterDelete = diagnosisDao.getDiagnosisById(diagnosis.id)
        assertNull(afterDelete)
    }
}
```

### End-to-End Testing

```kotlin
@RunWith(AndroidJUnit4::class)
@LargeTest
class DiagnosisE2ETest {
    
    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)
    
    @Test
    fun completeDiagnosisFlow() {
        // Navigate to diagnosis screen
        onView(withId(R.id.diagnose_button)).perform(click())
        
        // Select crop type
        onView(withId(R.id.crop_type_spinner)).perform(click())
        onData(allOf(`is`(instanceOf(String::class.java)), `is`("Cotton")))
            .perform(click())
        
        // Capture image (using test image)
        onView(withId(R.id.capture_button)).perform(click())
        
        // Wait for classification
        onView(withId(R.id.progress_indicator))
            .check(matches(isDisplayed()))
        
        // Verify results displayed
        onView(withId(R.id.disease_name))
            .check(matches(isDisplayed()))
        onView(withId(R.id.confidence_score))
            .check(matches(isDisplayed()))
        onView(withId(R.id.treatment_recommendations))
            .check(matches(isDisplayed()))
        
        // Verify saved to history
        pressBack()
        onView(withId(R.id.history_button)).perform(click())
        onView(withId(R.id.history_list))
            .check(matches(hasDescendant(withText(containsString("Cotton")))))
    }
}
```

### Performance Testing

```kotlin
class PerformanceTest {
    
    @Test
    fun testInferencePerformance() {
        val classifier = DiseaseClassifier(context)
        classifier.initialize()
        
        val testImages = loadTestImages(100)
        val inferenceTimes = mutableListOf<Long>()
        
        testImages.forEach { image ->
            val start = System.currentTimeMillis()
            classifier.classify(image)
            val duration = System.currentTimeMillis() - start
            inferenceTimes.add(duration)
        }
        
        val averageTime = inferenceTimes.average()
        val maxTime = inferenceTimes.maxOrNull() ?: 0L
        
        // Assert performance requirements
        assertTrue("Average inference time should be < 3000ms", averageTime < 3000)
        assertTrue("Max inference time should be < 5000ms", maxTime < 5000)
    }
    
    @Test
    fun testDatabaseQueryPerformance() = runBlocking {
        // Insert 100 diagnoses
        repeat(100) {
            diagnosisDao.insertDiagnosis(createTestDiagnosis())
        }
        
        // Measure query time
        val start = System.currentTimeMillis()
        val results = diagnosisDao.getAllDiagnoses(testUserId).first()
        val duration = System.currentTimeMillis() - start
        
        // Assert performance requirement
        assertTrue("Query should complete in < 500ms", duration < 500)
        assertEquals(100, results.size)
    }
}
```

### Security Testing

```kotlin
class SecurityTest {
    
    @Test
    fun testDataEncryption() {
        val sensitiveData = "user_phone_number"
        val encryptedPrefs = EncryptedSharedPreferences.create(
            context,
            "test_prefs",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
        
        // Write data
        encryptedPrefs.edit().putString("phone", sensitiveData).apply()
        
        // Read raw file - should be encrypted
        val rawData = File(context.filesDir, "../shared_prefs/test_prefs.xml")
            .readText()
        
        assertFalse("Data should be encrypted", rawData.contains(sensitiveData))
        
        // Read through API - should be decrypted
        val decrypted = encryptedPrefs.getString("phone", null)
        assertEquals(sensitiveData, decrypted)
    }
    
    @Test
    fun testTLSConfiguration() {
        val okHttpClient = OkHttpClient.Builder()
            .connectionSpecs(listOf(ConnectionSpec.MODERN_TLS))
            .build()
        
        val request = Request.Builder()
            .url("https://api.agriedge.com/health")
            .build()
        
        val response = okHttpClient.newCall(request).execute()
        
        // Verify TLS version
        val handshake = response.handshake
        assertNotNull(handshake)
        assertTrue(
            "Should use TLS 1.3 or 1.2",
            handshake!!.tlsVersion in listOf(TlsVersion.TLS_1_3, TlsVersion.TLS_1_2)
        )
    }
}
```

## Deployment Considerations

### Implementation Roadmap

**Phase 1: Hackathon MVP (2-3 weeks)**

Focus: Core diagnostic functionality with AWS AI services

- [ ] Android app with camera integration
- [ ] TFLite model integration for offline diagnosis
- [ ] Basic UI with Jetpack Compose
- [ ] AWS Lambda functions for auth and sync
- [ ] DynamoDB tables for user data and diagnoses
- [ ] S3 storage for images
- [ ] Amazon Bedrock integration for treatment recommendations
- [ ] Amazon Transcribe/Polly for voice interface
- [ ] Basic Amazon Q chatbot integration

**Phase 2: Beta Release (4-6 weeks)**

Focus: Market connectivity and enhanced features

- [ ] Beckn/ONDC integration for market search
- [ ] Transaction management
- [ ] Provider ratings and reviews
- [ ] Enhanced voice interface with command parsing
- [ ] Multi-language support (6 Indian languages)
- [ ] Offline sync queue implementation
- [ ] CloudFront CDN setup
- [ ] Amazon QuickSight dashboards for impact tracking
- [ ] Property-based testing implementation
- [ ] Security hardening (encryption, TLS)

**Phase 3: Production Scale (8-12 weeks)**

Focus: Scalability, reliability, and advanced features

- [ ] SageMaker pipelines for demand forecasting
- [ ] Advanced Amazon Q knowledge base with agricultural documents
- [ ] Performance optimization (caching, compression)
- [ ] Comprehensive monitoring and alerting
- [ ] Disaster recovery setup
- [ ] Load testing and capacity planning
- [ ] User onboarding and training materials
- [ ] Support infrastructure
- [ ] Analytics and reporting
- [ ] Continuous model improvement pipeline

### Android App Distribution

**Release Channels**:
1. **Internal Testing**: Team members only
2. **Closed Beta**: Selected farmers (100-500 users)
3. **Open Beta**: Public beta on Play Store
4. **Production**: Full release

**Release Checklist**:
- [ ] All property tests passing
- [ ] All unit tests passing
- [ ] Performance benchmarks met
- [ ] Security audit completed
- [ ] Localization verified for all languages
- [ ] ProGuard/R8 configuration tested
- [ ] APK size within limits (<150MB)
- [ ] Crash reporting configured (Firebase Crashlytics)
- [ ] Analytics configured (Firebase Analytics)
- [ ] Release notes prepared

### Backend Deployment (AWS)

**Infrastructure as Code (Terraform)**:

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"  # Mumbai
}

# Lambda functions
resource "aws_lambda_function" "auth_lambda" {
  filename      = "auth_lambda.zip"
  function_name = "agriedge-auth"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.lambda_handler"
  runtime       = "python3.11"
  timeout       = 30
  memory_size   = 512

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.users.name
      SNS_TOPIC_ARN  = aws_sns_topic.otp.arn
    }
  }
}

# DynamoDB tables
resource "aws_dynamodb_table" "users" {
  name           = "Users"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }

  attribute {
    name = "phone_number"
    type = "S"
  }

  global_secondary_index {
    name            = "PhoneNumberIndex"
    hash_key        = "phone_number"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = {
    Environment = "production"
    Application = "agriedge"
  }
}

# S3 buckets
resource "aws_s3_bucket" "images" {
  bucket = "agriedge-images"

  tags = {
    Environment = "production"
    Application = "agriedge"
  }
}

resource "aws_s3_bucket_versioning" "images_versioning" {
  bucket = aws_s3_bucket.images.id

  versioning_configuration {
    status = "Enabled"
  }
}

# CloudFront distribution
resource "aws_cloudfront_distribution" "cdn" {
  enabled = true
  comment = "AgriEdge CDN"

  origin {
    domain_name = aws_s3_bucket.images.bucket_regional_domain_name
    origin_id   = "S3-agriedge-images"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.oai.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-agriedge-images"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 86400
    max_ttl                = 31536000
    compress               = true
  }

  price_class = "PriceClass_200"

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = ["IN"]  # India only
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "api" {
  name          = "agriedge-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PUT", "DELETE"]
    allow_headers = ["*"]
  }
}
```

**Deployment Strategy**:
- **Blue-Green Deployment**: Zero-downtime deployments using Lambda aliases
- **Canary Releases**: Gradual rollout to subset of users (10% → 50% → 100%)
- **Rollback Plan**: Automated rollback on error rate threshold (>1%)

**CI/CD Pipeline (GitHub Actions)**:

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1
      
      - name: Run tests
        run: |
          python -m pytest tests/
      
      - name: Package Lambda functions
        run: |
          cd lambda
          zip -r auth_lambda.zip auth/
          zip -r sync_lambda.zip sync/
      
      - name: Deploy with Terraform
        run: |
          terraform init
          terraform plan
          terraform apply -auto-approve
      
      - name: Update Lambda functions
        run: |
          aws lambda update-function-code \
            --function-name agriedge-auth \
            --zip-file fileb://lambda/auth_lambda.zip
      
      - name: Run smoke tests
        run: |
          python scripts/smoke_tests.py
```

**Monitoring**:
- **Metrics**: Request rate, error rate, latency (p50, p95, p99)
- **Alerts**: 
  - Error rate > 1%
  - Latency p95 > 1s
  - DynamoDB throttling
  - Lambda concurrent executions > 80% of limit
- **Dashboards**: Real-time system health, user activity, API performance

### Disaster Recovery

**Backup Strategy**:
- **DynamoDB**: Point-in-time recovery enabled (35-day retention)
- **S3**: Versioning enabled, cross-region replication to ap-southeast-1 (Singapore)
- **RDS**: Automated daily backups, 30-day retention
- **Testing**: Monthly restore tests

**Recovery Procedures**:
- **RTO** (Recovery Time Objective): 4 hours
- **RPO** (Recovery Point Objective): 1 hour
- **Failover**: Multi-region setup with Route 53 health checks

**Multi-Region Architecture**:
```
Primary Region: ap-south-1 (Mumbai)
Secondary Region: ap-southeast-1 (Singapore)

Route 53 → Health Check → Primary Region (Active)
                       → Secondary Region (Standby)
```

