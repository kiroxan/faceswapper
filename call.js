const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function runSwap() {
  const form = new FormData();
  form.append('source', fs.createReadStream('./test/face.png'));  // Face to use
  form.append('target', fs.createReadStream('./test/target.png')); // Where to place the face
  
  // Optional: include mask if available
  if (fs.existsSync('./test/mask.png')) {
    form.append('mask', fs.createReadStream('./test/mask.png'));
  }
  
  try {
    const response = await axios.post(
      'http://localhost:7860/swap',
      form,
      {
        headers: form.getHeaders(),
        responseType: 'arraybuffer'
      }
    );
    
    fs.writeFileSync('./result.png', response.data);
    console.log('Face swap complete. Saved as result.png');
  } catch (error) {
    console.error('Error during face swap:', error);
  }
}

runSwap();